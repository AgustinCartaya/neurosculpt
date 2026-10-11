"""Helper to run the external FreeSurfer / FastSurfer command-line tools."""
import subprocess


def run_command(command, success_message, **kwargs):
    """Run ``command`` (list of str); print ``success_message`` or the tool's stderr. Return True on success."""
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode == 0:
        print(success_message)
        return True
    print("Error:", result.stderr)
    return False
