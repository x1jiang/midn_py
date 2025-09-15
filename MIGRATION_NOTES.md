# Migration Notes

## Deprecated Columns (jobs table)
The following columns are deprecated in favor of consolidated JSON `parameters` storage using config schemas (`config/*.json`):

- iteration_before_first_imputation
- iteration_between_imputations
- imputation_trials

They are still populated for backward compatibility but new logic reads/writes canonical values in `Job.parameters`.

### Planned Removal
1. Introduce Alembic migration script once all legacy rows are migrated:
   - Backfill `parameters` for any rows missing keys.
   - Drop the three deprecated columns.
2. Update Pydantic schemas and service layer to remove references.

## Parameter Schema System
Algorithm parameter forms are now generated dynamically from JSON schema files (e.g. `SIMI.json`, `SIMICE.json`). Each schema supports:
- `required`: list of mandatory fields.
- Property metadata: `minimum`, `default`, and UI hints like `ui:label`, `ui:placeholder`, `ui:arrayInput`.

## Adding a New Algorithm
1. Create `config/NEWALG.json` schema.
2. Add the algorithm name (upper-case) to `settings._ALG`.
3. (Optional) Provide runtime normalization logic in `alg_runtime.build_runtime_config` if special mapping needed.

### Inheriting from Existing (SIMI / SIMICE)
To create a variant that behaves like SIMI or SIMICE but adds or overrides parameters:
1. Set a `base` field in the new schema (e.g. `"base": "SIMI"`).
2. Only include new or overridden properties; base properties and required list will be merged automatically.
3. Runtime mapping will detect `base` and apply the same transformation logic as the base algorithm.
4. If you need extra runtime normalization, add conditional logic in `build_runtime_config` keyed on the new algorithm name.

## Alembic Draft (Pseudo-code)
```
from alembic import op
import sqlalchemy as sa

revision = 'drop_deprecated_job_cols'
down_revision = 'prev'

def upgrade():
    # Ensure parameters has migrated values (custom Python pre-step or here)
    with op.get_bind() as conn:
        rows = conn.execute(sa.text('SELECT id, parameters, iteration_before_first_imputation, iteration_between_imputations, imputation_trials FROM jobs')).fetchall()
        for r in rows:
            params = r.parameters or {}
            changed = False
            if r.iteration_before_first_imputation is not None and 'iteration_before_first_imputation' not in params:
                params['iteration_before_first_imputation'] = r.iteration_before_first_imputation; changed = True
            if r.iteration_between_imputations is not None and 'iteration_between_imputations' not in params:
                params['iteration_between_imputations'] = r.iteration_between_imputations; changed = True
            if r.imputation_trials is not None and 'imputation_trials' not in params:
                params['imputation_trials'] = r.imputation_trials; changed = True
            if changed:
                conn.execute(sa.text('UPDATE jobs SET parameters = :p WHERE id = :i'), {'p': sa.text(':p') , 'i': r.id})
    op.drop_column('jobs', 'iteration_before_first_imputation')
    op.drop_column('jobs', 'iteration_between_imputations')
    op.drop_column('jobs', 'imputation_trials')

def downgrade():
    op.add_column('jobs', sa.Column('iteration_before_first_imputation', sa.Integer()))
    op.add_column('jobs', sa.Column('iteration_between_imputations', sa.Integer()))
    op.add_column('jobs', sa.Column('imputation_trials', sa.Integer(), server_default='10'))
```

Adjust the parameter update SQL to use proper parameter binding for your dialect.
