import asyncio
import json
import numpy as np
from scipy import stats
import pandas as pd
import traceback

from ..websockets.connection_manager import ConnectionManager
from .. import models
from .job_status import JobStatusTracker

# Use uploaded SIMI central helpers
from algorithms.SIMI.SIMICentral import aggregate_ls_stats, gaussian_impute

class SIMIService:
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.jobs = {}
        self.site_to_job = {}
        self.job_status_tracker = JobStatusTracker()

    async def start_job(self, db_job: models.Job, central_data: pd.DataFrame):
        job_id = db_job.id
        
        # Create job status for tracking
        job_status = self.job_status_tracker.start_job(job_id)
        job_status.add_message(f"Starting {db_job.algorithm} job with {len(db_job.participants or [])} participants")
        
        # Decide method based on parameters (binary => logistic)
        # First check missing_spec, then fall back to parameters for is_binary
        is_binary = False
        if db_job.missing_spec and "is_binary" in db_job.missing_spec:
            is_binary = db_job.missing_spec.get("is_binary", False)
        elif db_job.parameters and "is_binary" in db_job.parameters:
            is_binary = db_job.parameters.get("is_binary", False)
            
        method = "logistic" if is_binary else "gaussian"
        print(f"Is Binary:\t{is_binary}")
        print(f"Selected method:\t{method}")
        self.jobs[job_id] = {
            "status": "starting",
            "participants": list(set(db_job.participants or [])),
            "connected_sites": [],
            "data": {},  # gaussian sufficient stats from remotes
            "central_data": central_data,
            "missing_spec": db_job.missing_spec or {},
            "method": method,
            # logistic buffers
            "logit_n_by_site": {},
            "logit_buffers": {},  # site_id -> { 'H': np.ndarray, 'g': np.ndarray, 'Q': float }
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
            self.job_status_tracker.complete_job(job_id)
            
        except Exception as e:
            error_details = traceback.format_exc()
            error_message = f"Job failed: {str(e)}\n{error_details}"
            self.job_status_tracker.fail_job(job_id, str(e))
            print(error_message)

    async def wait_for_participants(self, job_id: int):
        while len(set(self.jobs[job_id]["connected_sites"])) < len(set(self.jobs[job_id]["participants"])):
            await asyncio.sleep(1)

    async def handle_message(self, site_id: str, message: str):
        data = json.loads(message)
        job_id = data.get("job_id")

        if data.get("type") == "connect":
            if job_id not in self.jobs:
                return
            job = self.jobs[job_id]
            if site_id in job["participants"] and site_id not in job["connected_sites"]:
                job["connected_sites"].append(site_id)
                
                # Add status message
                self.job_status_tracker.add_message(job_id, f"Site {site_id} connected ({len(job['connected_sites'])}/{len(job['participants'])} connected)")
                
            self.site_to_job[site_id] = job_id
            
            # Always use the correct method based on job configuration
            method = job["method"]
            
            # Explicitly check if we should use logistic for binary data
            if "missing_spec" in job and job["missing_spec"].get("is_binary", False):
                method = "logistic"
            elif "parameters" in job and job["parameters"].get("is_binary", False):
                method = "logistic"
                
            # Instruct method to remote
            await self.manager.send_to_site(json.dumps({"type": "method", "method": method}), site_id)
            
            # Update the stored method in case it was modified
            job["method"] = method
            
            # Add status message for method instruction
            self.job_status_tracker.add_message(job_id, f"Instructed site {site_id} to use {method} method")
            return

        # Resolve job_id via mapping if absent
        if job_id is None:
            job_id = self.site_to_job.get(site_id)
            if job_id is None:
                return

        if job_id not in self.jobs:
            return

        j = self.jobs[job_id]
        msg_type = data.get("type")

        if msg_type == "data":  # gaussian sufficient stats
            j["data"][site_id] = {
                "n": float(data["n"]),
                "XX": np.array(data["XX"]),
                "Xy": np.array(data["Xy"]),
                "yy": float(data["yy"])
            }
            self.job_status_tracker.add_message(job_id, f"Received data from site {site_id} (n={float(data['n'])})")
            
        elif msg_type == "n":  # logistic: initial sample size
            j["logit_n_by_site"][site_id] = float(data["n"])
            self.job_status_tracker.add_message(job_id, f"Received sample size from site {site_id} (n={float(data['n'])})")
            
        elif msg_type in {"H", "g", "Q"}:  # logistic: iteration payloads
            if site_id not in j["logit_buffers"]:
                j["logit_buffers"][site_id] = {}
            if msg_type in {"H", "g"}:
                j["logit_buffers"][site_id][msg_type] = np.array(data[msg_type])
            else:
                j["logit_buffers"][site_id][msg_type] = float(data[msg_type])
                
            self.job_status_tracker.add_message(job_id, f"Received {msg_type} from site {site_id}")
            
            # Check if all participants have delivered H,g,Q for this iteration
            if j.get("logit_event") is not None:
                ready = all(
                    site in j["logit_buffers"] and
                    all(k in j["logit_buffers"][site] for k in ("H", "g", "Q"))
                    for site in j["participants"]
                )
                if ready:
                    self.job_status_tracker.add_message(job_id, "All sites reported. Proceeding with calculation.")
                    j["logit_event"].set()

    async def run_imputation(self, job_id: int, db_job: models.Job):
        job_info = self.jobs[job_id]
        central_data = job_info["central_data"]
        mvar = job_info["missing_spec"].get("target_column_index", 1) - 1
        
        # Double-check method based on is_binary flag to ensure consistency
        is_binary = False
        if "missing_spec" in job_info and "is_binary" in job_info["missing_spec"]:
            is_binary = job_info["missing_spec"]["is_binary"]
        elif "parameters" in db_job.__dict__ and db_job.parameters and "is_binary" in db_job.parameters:
            is_binary = db_job.parameters["is_binary"]
            
        method = "logistic" if is_binary else "gaussian"
        job_info["method"] = method  # Update the stored method
        
        self.job_status_tracker.add_message(job_id, f"Running with {method} method (binary={is_binary})")

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
            while len(job_info["data"]) < len(job_info["participants"]):
                await asyncio.sleep(0.5)

            # Aggregate using SIMI module helper
            n, XX, Xy, yy = aggregate_ls_stats(local_stats, list(job_info["data"].values()))

            # Gaussian imputation path
            imputed_df = gaussian_impute(central_data, mvar, n, XX, Xy, yy, M=10)
        else:
            # Logistic path: iterative exchange with remotes
            p = X.shape[1]
            beta = np.zeros(p)
            lam = 1e-3
            maxiter = 25
            Q_old = -np.inf
            vcov_mat = None

            for mode in range(1, maxiter + 1):
                # Reset iteration buffers and event
                job_info["logit_buffers"] = {}
                job_info["logit_event"] = asyncio.Event()

                # Send mode + beta to all sites
                send_payload = json.dumps({
                    'type': 'mode',
                    'mode': mode,
                    'beta': beta.tolist()
                })
                send_tasks = [
                    self.manager.send_to_site(send_payload, site_id)
                    for site_id in job_info["participants"]
                ]
                await asyncio.gather(*send_tasks)

                # Local (central) contributions
                xb = X @ beta
                pos_mask = xb > 0
                pr = np.empty_like(xb)
                pr[pos_mask] = 1.0 / (1.0 + np.exp(-xb[pos_mask]))
                pr[~pos_mask] = np.exp(xb[~pos_mask]) / (1.0 + np.exp(xb[~pos_mask]))
                pr = np.clip(pr, 1e-15, 1 - 1e-15)

                w = pr * (1 - pr)
                H_local = (X.T * w) @ X
                g_local = X.T @ (y - pr)
                Q_local = float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))

                # Wait for remotes to respond
                try:
                    await asyncio.wait_for(job_info["logit_event"].wait(), timeout=60)
                except asyncio.TimeoutError:
                    raise RuntimeError("Timeout waiting for logistic iteration responses from remotes")

                # Aggregate
                H_total = H_local.copy()
                g_total = g_local.copy()
                Q_total = Q_local
                for site_id in job_info["participants"]:
                    buf = job_info["logit_buffers"].get(site_id, {})
                    H_total += buf.get('H', 0)
                    g_total += buf.get('g', 0)
                    Q_total += buf.get('Q', 0.0)

                # Regularize
                H_reg = H_total + lam * np.eye(p)
                try:
                    delta_beta = np.linalg.solve(H_reg, g_total)
                except np.linalg.LinAlgError:
                    # Fall back to small gradient step
                    delta_beta = 0.01 * g_total

                # Limit step size
                step_norm = float(np.linalg.norm(delta_beta))
                if step_norm > 1.0:
                    delta_beta *= (1.0 / step_norm)

                beta = beta + delta_beta
                vcov_mat = np.linalg.pinv(H_reg)

                # Convergence checks
                if mode > 1 and np.linalg.norm(delta_beta) < 1e-4:
                    break
                if mode > 5 and Q_total < Q_old:
                    # Backtrack a bit if likelihood decreased
                    beta = beta - 0.5 * delta_beta
                Q_old = Q_total

            # Terminate remotes politely
            term_payload = json.dumps({'type': 'mode', 'mode': 0})
            term_tasks = [
                self.manager.send_to_site(term_payload, site_id)
                for site_id in job_info["participants"]
            ]
            await asyncio.gather(*term_tasks, return_exceptions=True)

            # Impute at central using final beta, vcov
            D_imp = central_data.values.copy()
            X_miss = np.delete(D_imp[miss, :], mvar, axis=1)

            # Multiple imputations: sample alpha from Normal(beta, vcov)
            M = 10
            imp_list = []
            cvcov = np.linalg.cholesky(vcov_mat)
            for _ in range(M):
                alpha = beta + cvcov.T @ np.random.normal(0, 1, size=p)
                xb_miss = X_miss @ alpha
                pos_mask = xb_miss > 0
                pr_miss = np.empty_like(xb_miss)
                pr_miss[pos_mask] = 1.0 / (1.0 + np.exp(-xb_miss[pos_mask]))
                pr_miss[~pos_mask] = np.exp(xb_miss[~pos_mask]) / (1.0 + np.exp(xb_miss[~pos_mask]))
                pr_miss = np.clip(pr_miss, 1e-8, 1 - 1e-8)
                D_i = D_imp.copy()
                D_i[miss, mvar] = np.random.binomial(1, pr_miss).astype(float)
                imp_list.append(D_i)

            imputed_df = pd.DataFrame(imp_list[0], columns=central_data.columns)

        # Persist to temp CSV for download
        imputed_dataset_path = f"imputed_data_{job_id}.csv"
        imputed_df.to_csv(imputed_dataset_path, index=False)

        db_job.imputed_dataset_path = imputed_dataset_path
        db_job.status = "completed"

        del self.jobs[job_id]
