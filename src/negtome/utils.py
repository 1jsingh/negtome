

def update_args(defaults: NegtomeArgs, args: Optional[Dict[str, Any]]) -> NegtomeArgs:
    # If no user arguments are provided, return the defaults
    if not args:
        return defaults
    
    # Update the defaults with user-provided arguments
    for key, value in args.items():
        if hasattr(defaults, key):
            setattr(defaults, key, value)
        else:
            raise ValueError(f"Invalid argument: {key}")
    
    return defaults
    

