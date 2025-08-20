"""
SIMI service implementation for central site using standardized job protocol.
"""

import asyncio
import datetime
import json
import numpy as np
import os
import pandas as pd
import zipfile
from typing import Dict, Any, List, Set, Optional

from common.algorithm.job_protocol import (
    ProtocolMessageType, create_message, JobStatus
)
from .federated_job_protocol_service import FederatedJobProtocolService
from .. import models
from ..websockets.connection_manager import ConnectionManager

# Import the SIMI algorithm
from algorithms.SIMI.simi_central import SIMICentralAlgorithm


class SIMIService(FederatedJobProtocolService):
    """
    Service for the SIMI algorithm on the central site.
    Uses standardized federated job protocol.
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
        
        # Initialize job state using protocol service
        await self.initialize_job_state(job_id, db_job)
        
        # Add SIMI-specific state
        self.jobs[job_id].update({
            "central_data": central_data,
            "missing_spec": db_job.missing_spec or {},
            "method": method,
            "data": {},  # Data from remotes
            "logit_buffers": {},
            "logit_event": None,
        })
        
        job_status.add_message(f"Using {method} method for imputation")
        job_status.add_message(f"Waiting for participants to connect: {', '.join(self.jobs[job_id]['participants'])}")
        
        # With standardized protocol, we don't need to manually wait for participants or start imputation
        # The protocol will handle connection management and trigger _handle_algorithm_start when ready
        self.add_status_message(job_id, f"Job {job_id}: Waiting for all participants to connect via standardized protocol")
    
    async def _handle_algorithm_message(self, site_id: str, job_id: int, message_type: str, data: Dict[str, Any]) -> None:
        """
        Handle algorithm-specific messages for SIMI.
        
        Args:
            site_id: ID of the remote site
            job_id: ID of the job
            message_type: Type of the message
            data: Message data
        """
        job = self.jobs[job_id]
        
        # Handle algorithm-specific data messages (Gaussian sufficient stats)
        if message_type == "data":
            print(f"🔍 SIMI: Processing data message from site {site_id} for job {job_id}")
            print(f"🔍 SIMI: Available jobs: {list(self.jobs.keys())}")
            print(f"🔍 SIMI: Job exists: {job_id in self.jobs}")
            
            if job_id not in self.jobs:
                print(f"❌ SIMI: Job {job_id} not found in jobs dict!")
                return
                
            job = self.jobs[job_id]
            print(f"🔍 SIMI: Job participants: {job.get('participants', [])}")
            print(f"🔍 SIMI: Current data keys: {list(job.get('data', {}).keys())}")
            
            job["data"][site_id] = {
                "n": float(data["n"]),
                "XX": np.array(data["XX"]),
                "Xy": np.array(data["Xy"]),
                "yy": float(data["yy"])
            }
            print(f"📊 SIMI: Received data from site {site_id} for job {job_id}")
            print(f"📈 SIMI: Current data count: {len(job['data'])}/{len(job['participants'])}")
            print(f"📋 SIMI: Sites with data: {list(job['data'].keys())}")
            print(f"📋 SIMI: Expected participants: {job['participants']}")
            self.add_status_message(job_id, f"Received data from site {site_id} (n={float(data['n'])})")
            
            # Check if all sites have sent their Gaussian data
            if len(job["data"]) >= len(job["participants"]):
                print(f"✅ SIMI: All sites have sent Gaussian data - starting imputation")
                self.add_status_message(job_id, f"All {len(job['participants'])} sites sent data - starting Gaussian imputation")
                
                # Start Gaussian imputation
                try:
                    # Find the db_job record to pass to run_imputation
                    from .. import models
                    from ..db import get_db
                    db = next(get_db())
                    db_job = db.query(models.Job).filter(models.Job.id == job_id).first()
                    
                    if db_job:
                        asyncio.create_task(self.run_imputation(job_id, db_job))
                        print(f"🚀 SIMI: Gaussian imputation task started for job {job_id}")
                    else:
                        print(f"❌ SIMI: Could not find database record for job {job_id}")
                        self.add_status_message(job_id, f"ERROR: Could not find job record for imputation")
                except Exception as e:
                    print(f"❌ SIMI: Error starting Gaussian imputation: {str(e)}")
                    self.add_status_message(job_id, f"ERROR: Failed to start imputation: {str(e)}")
        
        # Handle sample size messages (logistic regression)
        elif message_type == "n":
            if "logit_n_by_site" not in job:
                job["logit_n_by_site"] = {}
            job["logit_n_by_site"][site_id] = float(data["n"])
            self.add_status_message(job_id, f"Received sample size from site {site_id} (n={float(data['n'])})")
            print(f"📊 SIMI: Received sample size from site {site_id}: n={float(data['n'])}")
            
            # Check if all sites have sent their sample sizes
            expected_sites = set(job["participants"])
            received_sites = set(job["logit_n_by_site"].keys())
            print(f"📋 SIMI: Sample sizes received from: {received_sites}")
            print(f"📋 SIMI: Expected participants: {expected_sites}")
            
            if received_sites >= expected_sites:  # All sites have sent sample sizes
                print(f"✅ SIMI: All sites have sent sample sizes - starting logistic regression iterations")
                self.add_status_message(job_id, f"All {len(expected_sites)} sites sent sample sizes - starting logistic regression")
                
                # Start logistic regression iterations
                asyncio.create_task(self._start_logistic_regression(job_id))
        
        # Handle logistic regression iteration messages
        elif message_type in {"H", "g", "Q"}:
            if site_id not in job["logit_buffers"]:
                job["logit_buffers"][site_id] = {}
                
            if message_type in {"H", "g"}:
                job["logit_buffers"][site_id][message_type] = np.array(data[message_type])
            else:
                job["logit_buffers"][site_id][message_type] = float(data[message_type])
                
            self.add_status_message(job_id, f"Received {message_type} from site {site_id}")
            print(f"🔍 SIMI: Stored {message_type} from site {site_id}, buffers now: {list(job['logit_buffers'].keys())}")
            
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
    
    async def _handle_algorithm_start(self, job_id: int) -> None:
        """
        Handle SIMI algorithm start - send method to all sites.
        
        Args:
            job_id: ID of the job
        """
        job = self.jobs.get(job_id)
        if not job:
            return
        
        # Send method message to all connected sites
        method = job.get("method", "gaussian")
        print(f"🚀 SIMI: Starting algorithm with method: {method}")
        
        # Create method message
        method_message = create_message(
            ProtocolMessageType.METHOD,
            job_id=job_id,
            method=method
        )
        
        # Send to all connected sites
        for site_id in job["connected_sites"]:
            await self.manager.send_to_site(method_message, site_id)
            print(f"📤 SIMI: Sent method '{method}' to site {site_id}")
    
    async def _resend_setup_messages_for_reconnection(self, job_id: int, site_id: str) -> None:
        """
        Resend essential setup messages for a reconnecting site.
        
        Args:
            job_id: ID of the job
            site_id: ID of the reconnecting site
        """
        job = self.jobs.get(job_id)
        if not job:
            return
        
        print(f"🔄 SIMI: Resending setup messages for reconnecting site {site_id}")
        
        # Resend method message
        method = job.get("method", "gaussian")
        method_message = create_message(
            ProtocolMessageType.METHOD,
            job_id=job_id,
            method=method
        )
        await self.manager.send_to_site(method_message, site_id)
        print(f"📤 SIMI: Resent method '{method}' to reconnecting site {site_id}")
        
        # If the job is in computation phase, resend start_computation
        if job["status"] == JobStatus.ACTIVE.value:
            start_message = create_message(
                ProtocolMessageType.START_COMPUTATION,
                job_id=job_id
            )
            await self.manager.send_to_site(start_message, site_id)
            print(f"📤 SIMI: Resent start_computation to reconnecting site {site_id}")
            
            # For logistic method, if we're in the middle of iterations, send current state
            if method == "logistic":
                current_mode = job.get("current_mode", 1)
                print(f"🔄 SIMI: Job method is logistic, current_mode = {current_mode}")
                print(f"🔍 SIMI: Job keys: {list(job.keys())}")
                print(f"🔍 SIMI: Job status: {job.get('status')}")
                print(f"🔍 SIMI: Job participants: {job.get('participants', [])}")
                print(f"🔍 SIMI: Job ready_sites: {job.get('ready_sites', [])}")
                
                # Always send mode sync for logistic reconnections to help client decide
                mode_sync_message = create_message(
                    ProtocolMessageType.DATA,
                    job_id=job_id,
                    type="mode_sync",
                    current_mode=current_mode,
                    skip_initial=current_mode > 1,
                    message=f"Job is in mode {current_mode}, skip_initial={current_mode > 1}"
                )
                print(f"📤 SIMI: Sending mode sync message to {site_id}: {mode_sync_message}")
                await self.manager.send_to_site(mode_sync_message, site_id)
                print(f"📤 SIMI: Sent mode sync (mode {current_mode}, skip_initial={current_mode > 1}) to reconnecting site {site_id}")
                
                # If we're in an active iteration, also send the current mode message
                if current_mode > 0 and "current_beta" in job:
                    current_beta = job["current_beta"]
                    current_mode_message = create_message(
                        ProtocolMessageType.MODE,
                        job_id=job_id,
                        mode=current_mode,
                        beta=current_beta
                    )
                    print(f"📤 SIMI: Sending current mode {current_mode} message to reconnecting site {site_id}")
                    await self.manager.send_to_site(current_mode_message, site_id)
                    print(f"📤 SIMI: Sent current mode {current_mode} with beta to reconnecting site {site_id}")
                elif current_mode == 0:
                    # Send termination message
                    current_mode_message = create_message(
                        ProtocolMessageType.MODE,
                        job_id=job_id,
                        mode=0
                    )
                    print(f"📤 SIMI: Sending termination mode 0 to reconnecting site {site_id}")
                    await self.manager.send_to_site(current_mode_message, site_id)
                else:
                    print(f"⚠️ SIMI: No current_beta found in job for mode {current_mode}, reconnecting site will wait for next iteration")
        
        # Add site to ready sites if not already there (they were ready before)
        if site_id not in job["ready_sites"]:
            job["ready_sites"].append(site_id)
            print(f"✅ SIMI: Added reconnecting site {site_id} to ready sites")
    
    async def _start_logistic_regression(self, job_id: int) -> None:
        """
        Start the logistic regression iterations after all sites have sent sample sizes.
        
        Args:
            job_id: ID of the job
        """
        job_info = self.jobs[job_id]
        central_data = job_info["central_data"]
        mvar = job_info["missing_spec"].get("target_column_index", 1) - 1
        method = job_info["method"]
        
        print(f"🚀 SIMI: Starting logistic regression for job {job_id}")
        self.add_status_message(job_id, f"Job {job_id}: Starting logistic regression iterations")
        
        # Prepare local data
        miss = np.isnan(central_data.iloc[:, mvar].values)
        X = central_data.loc[~miss, :].drop(central_data.columns[mvar], axis=1).values
        y = central_data.loc[~miss, central_data.columns[mvar]].values
        
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
        
        # Main iteration loop (following R implementation pattern)
        iter_count = 0
        for iter_count in range(1, max_iter + 1):
            self.add_status_message(job_id, f"Job {job_id}: Starting logistic regression iteration {iter_count}/{max_iter}")
            
            # MODE 1: Newton-Raphson step (get H, g, Q from all sites)
            # Reset iteration buffers and event
            job_info["logit_buffers"] = {}
            job_info["logit_event"] = asyncio.Event()
            job_info["current_mode"] = 1
            job_info["current_beta"] = beta.tolist()
            
            print(f"📊 SIMI: Iteration {iter_count} - Mode 1 (Newton-Raphson)")
            
            # Send mode 1 + beta to all sites
            mode_message = create_message(
                ProtocolMessageType.MODE,
                job_id=job_id,
                mode=1,
                beta=beta.tolist()
            )
            
            for site_id in job_info["connected_sites"]:
                await self.manager.send_to_site(mode_message, site_id)
                print(f"📤 SIMI: Sent mode 1 message to site {site_id}")
                self.add_status_message(job_id, f"Sent mode 1 to site {site_id} - expecting H,g,Q response")
            
            # Local (central) contributions for mode 1
            xb = X @ beta
            pr = 1.0 / (1.0 + np.exp(-xb))
            pr = np.clip(pr, 1e-15, 1 - 1e-15)
            
            # Add regularization parameter
            N = len(y) + sum(job_info["logit_n_by_site"].values())  # Total sample size across all sites
            lam = 1e-3
            
            w = pr * (1 - pr)
            H_local = (X.T * w) @ X + lam * N * np.eye(p)
            g_local = X.T @ (y - pr) - lam * N * beta
            Q_local = float(np.sum(y * xb) + np.sum(np.log(1 - pr[pr < 0.5])) + 
                          np.sum(np.log(pr[pr >= 0.5]) - xb[pr >= 0.5]) - 
                          lam * N * np.sum(beta**2) / 2)
            
            # Wait for remotes to respond with H, g, Q
            try:
                self.add_status_message(job_id, f"Waiting for all sites to report H,g,Q (timeout: 60s)")
                await asyncio.wait_for(job_info["logit_event"].wait(), timeout=60.0)
                print(f"✅ SIMI: All sites responded for mode 1")
                
            except asyncio.TimeoutError:
                error_msg = f"Timeout waiting for sites to respond in mode 1 of iteration {iter_count}"
                print(f"❌ SIMI: {error_msg}")
                self.add_status_message(job_id, f"ERROR: {error_msg}")
                
                # Mark job as failed and notify all sites
                job_info["status"] = JobStatus.FAILED.value
                failure_message = create_message(
                    ProtocolMessageType.JOB_COMPLETED,
                    job_id=job_id,
                    status=JobStatus.FAILED.value,
                    message=error_msg
                )
                
                for site_id in job_info["connected_sites"]:
                    await self.manager.send_to_site(failure_message, site_id)
                
                return
            
            # Aggregate results from mode 1
            H_total = H_local.copy()
            g_total = g_local.copy()
            Q_total = Q_local
            
            for site_id in job_info["participants"]:
                buf = job_info["logit_buffers"].get(site_id, {})
                if 'H' in buf:
                    H_total += buf['H']
                if 'g' in buf:
                    g_total += buf['g']
                if 'Q' in buf:
                    Q_total += buf['Q']
            
            # Calculate search direction
            try:
                direction = np.linalg.solve(H_total, g_total)
            except np.linalg.LinAlgError:
                direction = 0.01 * g_total
            
            m = np.dot(direction, g_total)  # Gradient in search direction
            
            # LINE SEARCH using MODE 2 (following R implementation)
            step = 1.0
            line_search_iter = 0
            max_line_search = 20
            
            while line_search_iter < max_line_search:
                line_search_iter += 1
                nbeta = beta + step * direction
                
                # Check for convergence based on step size
                if np.max(np.abs(nbeta - beta)) < 1e-5:
                    print(f"📏 SIMI: Line search converged with step size {step:.6f}")
                    break
                
                # MODE 2: Line search evaluation (get only Q from all sites)
                job_info["logit_buffers"] = {}
                job_info["logit_event"] = asyncio.Event()
                job_info["current_mode"] = 2
                job_info["current_beta"] = nbeta.tolist()
                
                print(f"� SIMI: Line search step {line_search_iter} with step size {step:.6f}")
                
                # Send mode 2 + nbeta to all sites
                mode_message = create_message(
                    ProtocolMessageType.MODE,
                    job_id=job_id,
                    mode=2,
                    beta=nbeta.tolist()
                )
                
                for site_id in job_info["connected_sites"]:
                    await self.manager.send_to_site(mode_message, site_id)
                    print(f"� SIMI: Sent mode 2 (line search) to site {site_id}")
                
                # Local Q calculation for nbeta
                xb_new = X @ nbeta
                pr_new = 1.0 / (1.0 + np.exp(-xb_new))
                pr_new = np.clip(pr_new, 1e-15, 1 - 1e-15)
                
                Q_new_local = float(np.sum(y * xb_new) + np.sum(np.log(1 - pr_new[pr_new < 0.5])) + 
                                  np.sum(np.log(pr_new[pr_new >= 0.5]) - xb_new[pr_new >= 0.5]) - 
                                  lam * N * np.sum(nbeta**2) / 2)
                
                # Wait for remotes to respond with Q only
                try:
                    await asyncio.wait_for(job_info["logit_event"].wait(), timeout=60.0)
                    print(f"✅ SIMI: All sites responded for mode 2 (line search)")
                    
                except asyncio.TimeoutError:
                    error_msg = f"Timeout in line search mode 2 of iteration {iter_count}"
                    print(f"❌ SIMI: {error_msg}")
                    self.add_status_message(job_id, f"ERROR: {error_msg}")
                    
                    # Mark job as failed and notify all sites
                    job_info["status"] = JobStatus.FAILED.value
                    failure_message = create_message(
                        ProtocolMessageType.JOB_COMPLETED,
                        job_id=job_id,
                        status=JobStatus.FAILED.value,
                        message=error_msg
                    )
                    
                    for site_id in job_info["connected_sites"]:
                        await self.manager.send_to_site(failure_message, site_id)
                    
                    return
                
                # Aggregate Q values from mode 2
                Q_new_total = Q_new_local
                for site_id in job_info["participants"]:
                    buf = job_info["logit_buffers"].get(site_id, {})
                    if 'Q' in buf:
                        Q_new_total += buf['Q']
                
                # Check Armijo condition for line search
                if Q_new_total - Q_total > m * step / 2:
                    print(f"📈 SIMI: Line search successful: Q improved from {Q_total:.6f} to {Q_new_total:.6f}")
                    break
                
                # Reduce step size
                step = step / 2
                print(f"📉 SIMI: Reducing step size to {step:.6f}")
            
            # Check for convergence based on parameter change
            if np.max(np.abs(nbeta - beta)) < 1e-5:
                self.add_status_message(job_id, f"Job {job_id}: Convergence achieved after {iter_count} iterations")
                print(f"✅ SIMI: Converged after {iter_count} iterations")
                beta = nbeta
                break
            
            # Update beta for next iteration
            beta = nbeta
            
            # Track convergence
            delta_norm = np.linalg.norm(beta - (beta - step * direction))
            convergence_history.append(delta_norm)
            q_values.append(Q_new_total)
            
            # Log iteration results
            beta_str = " ".join([f"{b:.6f}" for b in beta])
            iter_detail = f"Iteration {iter_count}: Q = {Q_new_total:.6f}, beta = {beta_str}"
            self.add_status_message(job_id, iter_detail)
            print(f"ITERATION {iter_count}/{max_iter}: Q = {Q_new_total:.6f}, DELTA = {delta_norm:.6f}")
        
        # Send mode 0 to terminate all sites
        print(f"🏁 SIMI: Sending termination signal (mode 0) to all sites")
        job_info["current_mode"] = 0
        
        mode_message = create_message(
            ProtocolMessageType.MODE,
            job_id=job_id,
            mode=0
        )
        
        for site_id in job_info["connected_sites"]:
            await self.manager.send_to_site(mode_message, site_id)
            print(f"📤 SIMI: Sent mode 0 (termination) to site {site_id}")
        
        # Calculate final covariance matrix
        vcov_mat = np.linalg.pinv(H_total)
        
        # Store final results in job_info for imputation
        job_info["final_beta"] = beta
        job_info["final_vcov"] = vcov_mat
        job_info["convergence_history"] = convergence_history
        job_info["q_values"] = q_values
        
        print(f"🏁 SIMI: Logistic regression completed for job {job_id} after {iter_count} iterations")
        
        # Now trigger the imputation process
        print(f"🚀 SIMI: Starting imputation process for job {job_id}")
        try:
            # Find the db_job record to pass to run_imputation
            from .. import models
            from ..db import get_db
            db = next(get_db())
            db_job = db.query(models.Job).filter(models.Job.id == job_id).first()
            
            if db_job:
                await self.run_imputation(job_id, db_job)
                print(f"✅ SIMI: Imputation completed for job {job_id}")
            else:
                print(f"❌ SIMI: Could not find database record for job {job_id}")
                self.add_status_message(job_id, f"ERROR: Could not find job record for imputation")
        except Exception as e:
            print(f"❌ SIMI: Error during imputation: {str(e)}")
            self.add_status_message(job_id, f"ERROR: Imputation failed: {str(e)}")

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
            
            # All remote sites should have already provided their stats
            print(f"🔄 SIMI: Processing Gaussian data - Current data count: {len(job_info['data'])}/{len(job_info['participants'])}")
            print(f"📋 SIMI: Sites with data: {list(job_info['data'].keys())}")
            print(f"📋 SIMI: Expected participants: {job_info['participants']}")
            
            if len(job_info["data"]) < len(job_info["participants"]):
                print(f"❌ SIMI: Not all sites have sent data yet - {len(job_info['data'])}/{len(job_info['participants'])}")
                self.add_status_message(job_id, f"ERROR: Missing data from some sites")
                return
            
            self.add_status_message(job_id, f"Job {job_id}: Processing Gaussian data from all {len(job_info['participants'])} sites")
            
            # Make sure algorithm has the correct method before aggregating
            self.algorithm.method = method
            
            # Aggregate using the algorithm
            aggregated_data = await self.algorithm.aggregate_data(local_stats, list(job_info["data"].values()))
            
            # Run imputation
            imputed_dfs = await self.algorithm.impute(central_data, mvar, aggregated_data, method)
            
        else:  # Logistic method  
            # For logistic method, the results should already be available
            # since run_imputation is called directly from _start_logistic_regression
            
            print(f"🔄 SIMI: Checking for logistic regression results...")
            
            # Check if results are available
            if "final_beta" in job_info and "final_vcov" in job_info:
                # Results are available, create aggregated data for imputation
                beta = job_info["final_beta"]
                vcov_mat = job_info["final_vcov"]
                
                print(f"✅ SIMI: Found logistic regression results - beta shape: {np.array(beta).shape}")
                
                aggregated_data = {
                    "beta": np.array(beta),
                    "vcov": vcov_mat
                }
                
                # Run imputation
                imputed_dfs = await self.algorithm.impute(central_data, mvar, aggregated_data, method)
            else:
                print(f"❌ SIMI: Logistic regression results not found in job_info")
                print(f"🔍 SIMI: Available keys in job_info: {list(job_info.keys())}")
                self.add_status_message(job_id, f"ERROR: Logistic regression results not available")
                return
        
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
        
        # Create metadata about SIMI execution
        job_info = self.jobs.get(job_id, {})
        metadata = {
            'job_id': job_id,
            'algorithm': 'SIMI',
            'imputation_count': len(imputed_dfs),
            'method': method,
            'target_column': mvar,
            'is_binary': (method == "logistic"),  # Derive from method
            'total_sites': len(job_info.get('connected_sites', [])),
            'timestamp': timestamp,
            'note': 'SIMI results'
        }
        
        # Add convergence info if available
        job_info = self.jobs.get(job_id, {})
        if job_info.get("method") == "logistic" and "convergence_history" in job_info:
            history = job_info["convergence_history"]
            metadata['convergence_iterations'] = len(history)
            metadata['final_convergence_delta'] = history[-1] if history else None
        
        # Save metadata
        metadata_path = os.path.join(job_dir, 'simi_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create a zip file containing all CSVs and metadata
        import zipfile
        zip_path = os.path.join(results_dir, f"job_{job_id}_{timestamp}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for csv_file in csv_files:
                zipf.write(csv_file, os.path.basename(csv_file))
            # Add metadata file
            zipf.write(metadata_path, os.path.basename(metadata_path))
        
        # Create a relative path for the download URL
        relative_zip_path = os.path.join("static", "results", f"job_{job_id}_{timestamp}.zip")
        
        # Update job record in database using direct SQL to avoid session conflicts
        from ..db import get_db
        from sqlalchemy import text
        db = next(get_db())
        
        # Use a direct SQL update instead of modifying the object to avoid session conflicts
        db.execute(text(f"UPDATE jobs SET imputed_dataset_path = '{relative_zip_path}', status = 'completed' WHERE id = {job_id}"))
        db.commit()
        
        # Note: We're not using db.add(db_job) to avoid "already attached to session" errors
        
        print(f"🎉 SIMI: Job {job_id} completed successfully with {len(imputed_dfs)} imputations")
        print(f"📁 SIMI: Results available at: {relative_zip_path}")
        
        # Update job status with a specific message about imputation completion
        self.add_status_message(job_id, f"Job {job_id}: Imputation completed and data saved")
        
        # Add a summary of the convergence if we tracked it
        if job_info.get("method") == "logistic" and "convergence_history" in job_info:
            history = job_info["convergence_history"]
            final_iter = len(history)
            final_delta = history[-1] if history else "N/A"
            self.add_status_message(job_id, f"Job {job_id}: Logistic regression summary - {final_iter} iterations, final delta: {final_delta}")
        
        # Set the result to include the dataset path
        result = {"imputed_dataset_path": relative_zip_path}
        self.job_status_tracker.complete_job(job_id, result)
        
        # Use the standardized protocol for job completion notification
        await self.notify_job_completed(
            job_id, 
            result_path=relative_zip_path, 
            message='SIMI imputation completed successfully'
        )
        
        # We no longer wait 2 minutes since we've already sent the completion message
        # and we want to prevent unwanted reconnections
    
    async def _handle_algorithm_connection(self, site_id: str, job_id: int) -> None:
        """
        Handle algorithm-specific connection logic for SIMI.
        
        Args:
            site_id: ID of the remote site
            job_id: ID of the job
        """
        # Send method to the remote site
        job = self.jobs[job_id]
        method = job["method"]
        
        method_message = create_message(
            ProtocolMessageType.METHOD, 
            method=method
        )
        await self.manager.send_to_site(method_message, site_id)
        
        self.add_status_message(job_id, f"Instructed site {site_id} to use {method} method")
    
    def get_algorithm_name(self) -> str:
        """
        Get the name of the algorithm.
        
        Returns:
            Algorithm name
        """
        return "SIMI"
