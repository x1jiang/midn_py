from __future__ import annotations
from typing import Dict, Any, List
from .alg_config import get_algorithm_schema


def build_runtime_config(algorithm_name: str, db_job) -> Dict[str, Any]:
    algo = (algorithm_name or '').lower()
    raw_params: Dict[str, Any] = dict(db_job.parameters or {})
    config: Dict[str, Any] = {}
    # Detect base inheritance
    schema = get_algorithm_schema(algorithm_name)
    base = (schema or {}).get('base') or (schema or {}).get('BASE')
    base_lc = str(base).lower() if base else algo

    if algo == 'simi' or base_lc == 'simi':
        target_idx = raw_params.get('target_column_index')
        if target_idx is not None:
            try:
                t_int = int(target_idx)
                mvar = t_int - 1 if t_int > 0 else t_int
            except Exception:
                mvar = target_idx
            config['mvar'] = mvar
        if 'is_binary' in raw_params:
            config['method'] = 'logistic' if raw_params.get('is_binary') else 'Gaussian'
        elif 'method' in raw_params:
            config['method'] = raw_params['method']
        config['M'] = (raw_params.get('M') or raw_params.get('imputation_trials') or 1)
        for k, v in raw_params.items():
            if k not in {'target_column_index','is_binary','method','mvar','imputation_trials','M'}:
                config.setdefault(k, v)
    elif algo == 'simice' or base_lc == 'simice':
        def _int_list(val):
            if val is None:
                return []
            if isinstance(val, str):
                parts = [p.strip() for p in val.split(',') if p.strip()]
            else:
                parts = list(val)
            out: List[int] = []
            for p in parts:
                try:
                    iv = int(p)
                    out.append(iv - 1 if iv > 0 else iv)
                except Exception:
                    continue
            return out
        mvar_list = _int_list(raw_params.get('target_column_indexes') or raw_params.get('mvar'))
        if mvar_list:
            config['mvar'] = mvar_list
        if 'type_list' in raw_params and isinstance(raw_params['type_list'], list):
            type_list = raw_params['type_list']
        else:
            bin_list = raw_params.get('is_binary_list') or raw_params.get('is_binary')
            if isinstance(bin_list, list):
                type_list = ['logistic' if b else 'Gaussian' for b in bin_list]
            elif isinstance(bin_list, bool) and mvar_list:
                type_list = ['logistic' if bin_list else 'Gaussian'] * len(mvar_list)
            else:
                type_list = []
        if type_list:
            config['type_list'] = type_list
        iter_val = raw_params.get('iter_val') or raw_params.get('iteration_between_imputations')
        iter0_val = raw_params.get('iter0_val') or raw_params.get('iteration_before_first_imputation')
        if iter_val is not None:
            config['iter_val'] = iter_val
        if iter0_val is not None:
            config['iter0_val'] = iter0_val
        config['M'] = (raw_params.get('M') or raw_params.get('imputation_trials') or 1)
        skip = {'target_column_indexes','is_binary_list','is_binary','iteration_before_first_imputation','iteration_between_imputations','mvar','type_list','iter_val','iter0_val','imputation_trials','M'}
        for k, v in raw_params.items():
            if k in skip:
                continue
            config.setdefault(k, v)
    else:
        config = raw_params
    return config
