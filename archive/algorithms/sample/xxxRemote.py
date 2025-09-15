# Remote-side logic for the sample algorithm

def run_remote(data, params):
    """
    This function is executed on the remote site.
    """
    print("Running remote logic")
    # Perform computations on local data
    # Return aggregates/partials to the central server
    print("Remote logic finished")
    return {"result": "some_aggregate"}
