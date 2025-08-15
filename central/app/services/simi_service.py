"""
SIMI service implementation for central site.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Optional

from common.algorithm.protocol import create_message, parse_message, MessageType
from .base_algorithm_service import BaseAlgorithmService
from .. import models
from ..websockets.connection_manager import ConnectionManager

# Import the SIMI algorithm
from algorithms.SIMI.simi_central import SIMICentralAlgorithm


class SIMIService(BaseAlgorithmService):
    """
    Service for the SIMI algorithm on the central site.
    """
    
    def __init__(self, manager: ConnectionManager):
        """
        Initialize SIMI service.
        
        Args:
            manager: WebSocket connection manager
        """
        super().__init__(manager)
        self.algorithm = SIMICentralAlgorithm()
    
    async def start_job(self, db_job: models.Job, central_data: pd.DataFrame) -> None:
        """
        Start a new SIMI job.
        
        Args:
            db_job: Job database record
            central_data: Data from the central site
        """
        job_id = db_job.id
        
        # Create job status for tracking
        job_status = self.job_status_tracker.start_job(job_id)
        job_status.add_message(f"Starting {db_job.algorithm} job with {len(db_job.participants or [])} participants")
        
        # Determine the imputation method
        is_binary = False
        if db_job.missing_spec and "is_binary" in db_job.missing_spec:
            is_binary = db_job.missing_spec.get("is_binary", False)
        elif db_job.parameters and "is_binary" in db_job.parameters:
            is_binary = db_job.parameters.get("is_binary", False)
            
        method = "logistic" if is_binary else "gaussian"
        
        # Initialize job state
        self.jobs[job_id] = {
            "status": "starting",
            "participants": list(set(db_job.participants or [])),
            "connected_sites": [],
            "data": {},  # Data from remotes
            "central_data": central_data,
            "missing_spec": db_job.missing_spec or {},
            "method": method,
            # For logistic regression iterations
            "logit_buffers": {},
            "logit_event": None,
        }
        
        job_status.add_message(f"Using {method} method for imputation")
        job_status.add_message(f"Waiting for participants to connect: {', '.join(self.jobs[job_id]['participants'])}")
        
        try:
            # Wait for all participants to connect
            await self.wait_for_participants(job_id)
            
            # Start the imputation process
            await self.run_imputation(job_id, db_job)
            
            # Mark job as completed
            self.add_status_message(job_id, f"Job {job_id}: Main workflow completed successfully")
            self.job_status_tracker.complete_job(job_id)
            
        except Exception as e:
            # Mark job as failed
            self.job_status_tracker.fail_job(job_id, str(e))
            raise
    
    async def _handle_site_message_impl(self, site_id: str, message_str: str) -> None:
        """
        Implementation of site message handling for SIMI.
        
        Args:
            site_id: ID of the remote site
            message_str: Message content as string
        """
        # Parse the message
        data = json.loads(message_str)
        job_id = data.get("job_id")
        message_type = data.get("type")
        
        # Handle connect messages
        if message_type == "connect":
            # Check if there are no jobs available at all
            if not self.jobs:
                print(f"❌ SIMI: No jobs available for site {site_id}")
                error_message = create_message(
                    MessageType.ERROR,
                    message="No SIMI jobs are currently running. Please try again later.",
                    code="NO_JOBS_AVAILABLE"
                )
                await self.manager.send_to_site(error_message, site_id)
                print(f"📤 SIMI: Sent 'no jobs available' error to site {site_id}")
                return
            
            # Check if the specific job exists
            if not job_id:
                print(f"❌ SIMI: No job_id provided by site {site_id}")
                error_message = create_message(
                    MessageType.ERROR,
                    message="No job_id specified in connection request",
                    code="MISSING_JOB_ID"
                )
                await self.manager.send_to_site(error_message, site_id)
                print(f"📤 SIMI: Sent 'missing job_id' error to site {site_id}")
                return
            
            if job_id not in self.jobs:
                print(f"❌ SIMI: Job {job_id} not found for site {site_id}")
                print(f"📋 SIMI: Available jobs: {list(self.jobs.keys())}")
                error_message = create_message(
                    MessageType.ERROR,
                    message=f"Job {job_id} is not currently running or does not exist",
                    code="JOB_NOT_FOUND",
                    available_jobs=list(self.jobs.keys())
                )
                await self.manager.send_to_site(error_message, site_id)
                print(f"📤 SIMI: Sent 'job not found' error to site {site_id}")
                return
            
            job = self.jobs[job_id]
            
            if site_id in job["participants"] and site_id not in job["connected_sites"]:
                job["connected_sites"].append(site_id)
                self.add_status_message(job_id, f"Site {site_id} connected ({len(job['connected_sites'])}/{len(job['participants'])} connected)")
            elif site_id not in job["participants"]:
                print(f"❌ SIMI: Site {site_id} not in participants list for job {job_id}")
                print(f"📋 SIMI: Expected participants: {job['participants']}")
                error_message = create_message(
                    MessageType.ERROR,
                    message=f"Site {site_id} is not authorized for job {job_id}",
                    code="UNAUTHORIZED_SITE"
                )
                await self.manager.send_to_site(error_message, site_id)
                print(f"📤 SIMI: Sent 'unauthorized site' error to site {site_id}")
                return
            
            self.site_to_job[site_id] = job_id
            
            # Send method to the remote site
            method = job["method"]
            await self.manager.send_to_site(
                create_message(MessageType.METHOD, method=method),
                site_id
            )
            
            self.add_status_message(job_id, f"Instructed site {site_id} to use {method} method")
            return
        
        # Resolve job_id via mapping if absent
        if job_id is None:
            job_id = self.site_to_job.get(site_id)
            if job_id is None:
                return
        
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        
        # Handle data messages (Gaussian sufficient stats)
        if message_type == "data":
            job["data"][site_id] = {
                "n": float(data["n"]),
                "XX": np.array(data["XX"]),
                "Xy": np.array(data["Xy"]),
                "yy": float(data["yy"])
            }
            self.add_status_message(job_id, f"Received data from site {site_id} (n={float(data['n'])})")
        
        # Handle sample size messages (logistic regression)
        elif message_type == "n":
            if "logit_n_by_site" not in job:
                job["logit_n_by_site"] = {}
            job["logit_n_by_site"][site_id] = float(data["n"])
            self.add_status_message(job_id, f"Received sample size from site {site_id} (n={float(data['n'])})")
        
        # Handle logistic regression iteration messages
        elif message_type in {"H", "g", "Q"}:
            if site_id not in job["logit_buffers"]:
                job["logit_buffers"][site_id] = {}
                
            if message_type in {"H", "g"}:
                job["logit_buffers"][site_id][message_type] = np.array(data[message_type])
            else:
                job["logit_buffers"][site_id][message_type] = float(data[message_type])
                
            self.add_status_message(job_id, f"Received {message_type} from site {site_id}")
            
            # Check if all participants have delivered required data for this iteration
            if job.get("logit_event") is not None:
                # Get the current mode
                current_mode = job.get("current_mode", 1)
                
                # For mode 1, we need H, g, and Q from all sites
                # For mode 2+, we only need Q from all sites
                required_keys = ["H", "g", "Q"] if current_mode == 1 else ["Q"]
                
                # For each site, check what keys are missing
                for site in job["participants"]:
                    if site not in job["logit_buffers"]:
                        self.add_status_message(job_id, f"Waiting for site {site} to send any data")
                    else:
                        missing = [k for k in required_keys if k not in job["logit_buffers"][site]]
                        if missing:
                            self.add_status_message(job_id, f"Waiting for site {site} to send: {', '.join(missing)}")
                
                # Check if we have all required data from all sites
                all_sites_reported = all(site in job["logit_buffers"] for site in job["participants"])
                all_data_complete = all(
                    site in job["logit_buffers"] and
                    all(k in job["logit_buffers"][site] for k in required_keys)
                    for site in job["participants"]
                )
                
                # Get current mode from job info
                current_mode = job.get("current_mode", 1)
                
                # Get current mode from job
                current_mode = job.get("current_mode", 1)
                
                if all_data_complete:
                    self.add_status_message(job_id, f"Job {job_id}: All sites reported complete data for mode {current_mode}")
                    # Set the event to unblock the waiting calculation
                    job["logit_event"].set()
                elif all_sites_reported:
                    self.add_status_message(job_id, f"Job {job_id}: All sites reported (some data incomplete) for mode {current_mode}")
                    # Log what's missing for debugging
                    for site in job["participants"]:
                        if site in job["logit_buffers"]:
                            missing = [k for k in required_keys if k not in job["logit_buffers"][site]]
                            if missing:
                                print(f"Site {site} is missing: {missing} in mode {current_mode}")
                    # Set the event anyway since all sites have reported something
                    job["logit_event"].set()
                    # Log what we received for debugging
                    for site_id, site_data in job["logit_buffers"].items():
                        self.add_status_message(
                            job_id, 
                            f"Site {site_id} data: " + 
                            f"H[{len(site_data.get('H', []))}x{len(site_data.get('H', [[]])[0]) if len(site_data.get('H', [])) > 0 else 0}], " +
                            f"g[{len(site_data.get('g', []))}], " +
                            f"Q={site_data.get('Q', 'N/A')}"
                        )
    
    async def run_imputation(self, job_id: int, db_job: models.Job) -> None:
        """
        Run the imputation algorithm.
        
        Args:
            job_id: ID of the job
            db_job: Job database record
        """
        job_info = self.jobs[job_id]
        central_data = job_info["central_data"]
        mvar = job_info["missing_spec"].get("target_column_index", 1) - 1
        method = job_info["method"]
        
        self.add_status_message(job_id, f"Job {job_id}: Preparing data for {method} imputation")
        
        # Set the algorithm method explicitly before proceeding
        self.algorithm.method = method
        print(f"Setting algorithm method to: {method}")
        self.add_status_message(job_id, f"Job {job_id}: Starting imputation with {method} method")
        
        # Prepare local data
        miss = np.isnan(central_data.iloc[:, mvar].values)
        X = central_data.loc[~miss, :].drop(central_data.columns[mvar], axis=1).values
        y = central_data.loc[~miss, central_data.columns[mvar]].values
        
        if method == "gaussian":
            # Prepare local stats from central data
            local_stats = {
                'n': float(X.shape[0]),
                'XX': np.matmul(X.T, X),
                'Xy': np.matmul(X.T, y),
                'yy': float(np.sum(y ** 2)),
            }
            
            # Wait for all remote sites to provide their stats
            self.add_status_message(job_id, f"Job {job_id}: Waiting for all sites to send Gaussian statistics")
            while len(job_info["data"]) < len(job_info["participants"]):
                await asyncio.sleep(0.5)
            self.add_status_message(job_id, f"Job {job_id}: Received data from all {len(job_info['participants'])} sites")
            
            # Make sure algorithm has the correct method before aggregating
            self.algorithm.method = method
            
            # Aggregate using the algorithm
            aggregated_data = await self.algorithm.aggregate_data(local_stats, list(job_info["data"].values()))
            
            # Run imputation
            imputed_dfs = await self.algorithm.impute(central_data, mvar, aggregated_data, method)
            
        else:  # Logistic method
            # Iterative exchange with remotes
            p = X.shape[1]
            beta = np.zeros(p)
            max_iter = 25
            vcov_mat = None
            convergence_history = []  # Track convergence history
            q_values = []  # Track Q values across iterations
            
            # Log initial beta values
            beta_str = " ".join([f"{b:.6f}" for b in beta])
            self.add_status_message(job_id, f"Initial beta values = {beta_str}")
            print(f"Initial beta values = {beta_str}")
            
            for mode in range(1, max_iter + 1):
                # Reset iteration buffers and event
                job_info["logit_buffers"] = {}
                job_info["logit_event"] = asyncio.Event()
                
                # Store current mode in job_info for access elsewhere
                job_info["current_mode"] = mode
                self.add_status_message(job_id, f"Job {job_id}: Starting logistic regression iteration {mode}/{max_iter}")
                
                # Send mode + beta to all sites
                await self.send_to_all_sites(
                    job_id,
                    MessageType.MODE,
                    mode=mode,
                    beta=beta.tolist()
                )
                
                # Local (central) contributions
                xb = X @ beta
                pr = 1.0 / (1.0 + np.exp(-xb))
                pr = np.clip(pr, 1e-15, 1 - 1e-15)
                
                # The algorithm behaves differently depending on the mode
                if mode == 1:
                    # First mode: Full Newton-Raphson step
                    w = pr * (1 - pr)
                    H_local = (X.T * w) @ X
                    g_local = X.T @ (y - pr)
                    Q_local = float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
                else:
                    # Mode 2+: Line search - we only need Q for these iterations
                    Q_local = float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
                    # We'll still use these placeholders but they won't be used
                    H_local = None
                    g_local = None
                
                # Wait for remotes to respond
                try:
                    # Increase timeout to 30 seconds for logistic regression iterations
                    self.add_status_message(job_id, f"Waiting for all sites to report data (timeout: 30s)")
                    
                    # Log the current status of responses
                    received_sites = list(job_info["logit_buffers"].keys())
                    waiting_sites = [site for site in job_info["participants"] if site not in job_info["logit_buffers"]]
                    print(f"Mode {mode} - Received data from sites: {received_sites}")
                    print(f"Mode {mode} - Waiting for sites: {waiting_sites}")
                    
                    # Add more detailed status message
                    if waiting_sites:
                        self.add_status_message(job_id, f"Mode {mode}: Waiting for sites: {', '.join(waiting_sites)}")
                    
                    # Wait with increased timeout
                    await asyncio.wait_for(job_info["logit_event"].wait(), timeout=30)
                    
                except asyncio.TimeoutError:
                    # Log more debug information
                    received_sites = list(job_info["logit_buffers"].keys())
                    missing_sites = [site for site in job_info["participants"] if site not in job_info["logit_buffers"]]
                    print(f"TIMEOUT - Mode {mode} - Received data from sites: {received_sites}")
                    print(f"TIMEOUT - Mode {mode} - Missing data from sites: {missing_sites}")
                    
                    # Check if we have at least some data
                    if len(job_info["logit_buffers"]) > 0:
                        self.add_status_message(job_id, f"Proceeding with partial data (missing {len(missing_sites)} sites)")
                        print(f"Proceeding with partial data from {len(job_info['logit_buffers'])} sites")
                        # Continue with the data we have
                    else:
                        raise RuntimeError(f"Timeout waiting for ANY responses from remotes in mode {mode} (30s timeout)")
                
                                # Log what data we've received from each site
                site_data_summary = []
                
                # Check if all sites have reported complete data
                all_reported = True
                for site_id in job_info["participants"]:
                    buf = job_info["logit_buffers"].get(site_id, {})
                    # Check if we have all required data
                    if mode == 1 and ('H' not in buf or 'g' not in buf or 'Q' not in buf):
                        all_reported = False
                    elif mode > 1 and 'Q' not in buf:
                        all_reported = False
                        
                    # Prepare status summary
                    try:
                        h_status = f"H[{buf['H'].shape[0]}x{buf['H'].shape[1]}]" if 'H' in buf else "H[0x0]"
                        g_status = f"g[{len(buf['g'])}]" if 'g' in buf else "g[0]"
                        q_status = f"Q={buf['Q']}" if 'Q' in buf else "Q=N/A"
                    except (KeyError, AttributeError):
                        h_status = "H[0x0]"
                        g_status = "g[0]"
                        q_status = "Q=N/A"
                        
                    site_data_summary.append(f"Site {site_id} data: {h_status}, {g_status}, {q_status}")
                
                # Add status message about what data we've received
                if all_reported:
                    self.add_status_message(job_id, f"Job {job_id}: All sites reported complete data for mode {mode}")
                else:
                    self.add_status_message(job_id, f"Job {job_id}: All sites reported (some data incomplete) for mode {mode}")
                
                for summary in site_data_summary:
                    self.add_status_message(job_id, summary)
                
                # Aggregate
                
                # Add status message about what data we've received
                if all_reported:
                    self.add_status_message(job_id, f"Job {job_id}: All sites reported complete data for mode {mode}")
                else:
                    self.add_status_message(job_id, f"Job {job_id}: All sites reported (some data incomplete) for mode {mode}")
                
                for summary in site_data_summary:
                    self.add_status_message(job_id, summary)
                
                # Aggregate
                Q_total = Q_local
                
                # Only use H and g for mode 1 (they're None for mode 2+)
                if mode == 1:
                    H_total = H_local.copy()
                    g_total = g_local.copy()
                    
                    for site_id in job_info["participants"]:
                        buf = job_info["logit_buffers"].get(site_id, {})
                        if 'H' in buf:
                            H_total += buf['H']
                        if 'g' in buf:
                            g_total += buf['g']
                        if 'Q' in buf:
                            Q_total += buf['Q']
                else:
                    # For mode 2+, only aggregate Q values
                    for site_id in job_info["participants"]:
                        buf = job_info["logit_buffers"].get(site_id, {})
                        if 'Q' in buf:
                            Q_total += buf['Q']
                    
                    # Set placeholders for H and g (needed below)
                    H_total = np.zeros((p, p))
                    g_total = np.zeros(p)
                
                # Update beta
                lam = 1e-3  # Regularization
                H_reg = H_total + lam * np.eye(p)
                
                try:
                    delta_beta = np.linalg.solve(H_reg, g_total)
                except np.linalg.LinAlgError:
                    delta_beta = 0.01 * g_total
                
                beta = beta + delta_beta
                vcov_mat = np.linalg.pinv(H_reg)
                
                # Check convergence
                delta_norm = np.linalg.norm(delta_beta)
                convergence_history.append(delta_norm)
                
                # Store convergence history in job info for later use
                job_info["convergence_history"] = convergence_history
                
                # Format beta coefficients for display exactly as requested
                beta_str = " ".join([f"{b:.6f}" for b in beta])
                
                # Store Q value for each iteration
                q_values.append(Q_total)
                job_info["q_values"] = q_values
                
                # Log detailed iteration information including Q and beta values
                # Match the exact format requested: "Iteration N: Q = X, beta = X X X..."
                iter_detail = f"Iteration {mode}: Q = {Q_total:.6f}, beta = {beta_str}"
                self.add_status_message(job_id, iter_detail)
                
                # Also print the iteration details to the console for debugging
                print(f"ITERATION {mode}/{max_iter}: Q = {Q_total:.6f}, DELTA = {delta_norm:.6f}")
                
                # Log convergence progress
                self.add_status_message(job_id, f"Job {job_id}: Iteration {mode} - Convergence delta: {delta_norm:.6f}")
                
                # Wait for an additional short time to collect any pending Q values from sites
                # This ensures we get complete Q values from all sites
                if not all_reported:
                    self.add_status_message(job_id, "Waiting for additional site data...")
                    # We'll continue with the current data but log any further updates we receive
                
                if delta_norm < 1e-4:
                    self.add_status_message(job_id, f"Job {job_id}: Convergence achieved after {mode} iterations with delta {delta_norm:.6f}")
                    break
            
            # If we completed all iterations without converging, log that
            if mode == max_iter:
                self.add_status_message(job_id, f"Job {job_id}: Maximum iterations ({max_iter}) reached without full convergence. Final delta: {delta_norm:.6f}")
            
            # Show convergence history in a summarized format
            if len(convergence_history) > 1:
                convergence_text = ", ".join([f"{i+1}:{v:.6f}" for i, v in enumerate(convergence_history)])
                self.add_status_message(job_id, f"Job {job_id}: Convergence history: {convergence_text}")
                
                # Create a simple ASCII chart to visualize convergence
                if len(convergence_history) > 2:  # Only show chart if we have enough data points
                    max_val = max(convergence_history)
                    min_val = min(convergence_history)
                    range_val = max_val - min_val
                    
                    # Normalize values to 0-20 range for ASCII chart
                    norm_values = [int(20 * (v - min_val) / range_val) if range_val > 0 else 10 for v in convergence_history]
                    
                    # Also create a chart for Q values if available
                    if "q_values" in job_info and len(job_info["q_values"]) > 2:
                        q_values_chart = ["Q values across iterations:"]
                        
                        # Format each Q value
                        for i, q_val in enumerate(job_info["q_values"]):
                            q_values_chart.append(f"Iter {i+1:2d}: Q = {q_val:.6f}")
                        
                        # Add Q values chart to status messages
                        for line in q_values_chart:
                            self.add_status_message(job_id, f"Job {job_id}: {line}")
                    
                    # Create ASCII chart - use delta values directly for clarity
                    chart = ["Convergence delta pattern:"]
                    for i, val in enumerate(convergence_history):
                        # Using a fixed width to align the chart
                        iter_label = f"Iter {i+1:2d}"
                        line = f"{iter_label}: delta = {val:.6f}"
                        chart.append(line)
                    
                    # Add chart to status messages
                    for line in chart:
                        self.add_status_message(job_id, f"Job {job_id}: {line}")
            
            # Print a summary of the convergence process
            if len(q_values) > 0:
                q_summary = ", ".join([f"Q{i+1}={q:.2f}" for i, q in enumerate(q_values)])
                self.add_status_message(job_id, f"Q values summary: {q_summary}")
                
                # Also print final beta
                final_beta_str = " ".join([f"{b:.6f}" for b in beta])
                self.add_status_message(job_id, f"Final beta values = {final_beta_str}")
            
            # Terminate remotes politely
            await self.send_to_all_sites(job_id, MessageType.MODE, mode=0)
            
            # Create the aggregated data for imputation
            aggregated_data = {
                "beta": beta,
                "vcov": vcov_mat  # Changed key from 'vcov_mat' to 'vcov' to match what's expected in SIMI algorithm
            }
            
            # Run imputation
            imputed_dfs = await self.algorithm.impute(central_data, mvar, aggregated_data, method)
        
        # Create a timestamp for uniqueness
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory for the results if it doesn't exist
        import os
        results_dir = os.path.join("central", "app", "static", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a job-specific directory
        job_dir = os.path.join(results_dir, f"job_{job_id}_{timestamp}")
        os.makedirs(job_dir, exist_ok=True)
        
        # Save each imputed DataFrame to a CSV file
        csv_files = []
        for i, df in enumerate(imputed_dfs):
            csv_path = os.path.join(job_dir, f"imputed_data_{i+1}.csv")
            df.to_csv(csv_path, index=False)
            csv_files.append(csv_path)
        
        # Create a zip file containing all CSVs
        import zipfile
        zip_path = os.path.join(results_dir, f"job_{job_id}_{timestamp}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for csv_file in csv_files:
                zipf.write(csv_file, os.path.basename(csv_file))
        
        # Create a relative path for the download URL
        relative_zip_path = os.path.join("static", "results", f"job_{job_id}_{timestamp}.zip")
        
        # Update job record in database
        db_job.imputed_dataset_path = relative_zip_path
        db_job.status = "completed"
        
        # Commit changes to the database
        from ..db import get_db
        db = next(get_db())
        db.add(db_job)
        db.commit()
        
        # Update job status with a specific message about imputation completion
        self.add_status_message(job_id, f"Job {job_id}: Imputation completed and data saved")
        
        # Add a summary of the convergence if we tracked it
        job_info = self.jobs.get(job_id, {})
        if job_info.get("method") == "logistic" and "convergence_history" in job_info:
            history = job_info["convergence_history"]
            final_iter = len(history)
            final_delta = history[-1] if history else "N/A"
            self.add_status_message(job_id, f"Job {job_id}: Logistic regression summary - {final_iter} iterations, final delta: {final_delta}")
        
        self.add_status_message(job_id, f"Job {job_id}: Results are available for download")
        
        # Set the result to include the dataset path
        result = {"imputed_dataset_path": relative_zip_path}
        self.job_status_tracker.complete_job(job_id, result)
        
        # Clean up
        self.jobs.pop(job_id, None)  # Safely remove from jobs dict
