from stt import LinVoice
import sys

def run_test():
    print("--- LinAgent STT Library Test ---")
    
    try:
        # 1. Test Initialization
        print("[1/3] Loading model into memory...")
        voice_lib = LinVoice(model_path="../models/distil-large-v3")
        print("Success: Model loaded.")

        # 2. Test Recording & Transcription
        print("[2/3] Speak a test phrase now (3s)...")
        result = voice_lib.listen(duration=3)
        
        # 3. Verify Output
        if result:
            print(f"[3/3] Library Output: '{result}'")
            print("\nRESULT: STT Library is working perfectly.")
        else:
            print("[3/3] Library returned empty string (no speech detected).")
            print("\nRESULT: Library is functional, but didn't catch audio.")

    except Exception as e:
        print(f"\nERROR: Library test failed with: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
