try:
    print("Attempting to import ml_optimizer...")
    from ml_optimizer import get_ml_system
    print("Import successful!")
    system = get_ml_system()
    print("System status:", system.get_ml_system_status())
except Exception as e:
    import traceback
    traceback.print_exc()
