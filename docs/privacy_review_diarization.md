# Privacy Review: app/services/diarization.py

This file implements the speaker diarization functionality using the `pyannote.audio` library.

## Analysis

1.  **Purpose:** Identifies different speakers in an audio file and provides timestamps for each speaker's segments.
2.  **Input:** Receives the path to an audio file (`file_path`, typically the temporary processed file from `_preprocess_audio` or the original if preprocessing fails/is skipped), speaker count parameters, and optionally a `task_id`.
3.  **Processing & Temporary Files:**
    *   **Preprocessing (`_preprocess_audio`):**
        *   Uses `ffmpeg` to normalize, resample, and convert the input audio to mono WAV format.
        *   Creates a temporary processed file (e.g., `processed_some-uuid.wav`) in the `settings.RESULTS_DIR`.
        *   Returns the path to this temporary file and a flag indicating it's temporary.
    *   **Chunking (`_split_audio_chunks`):**
        *   If the audio duration exceeds a configured threshold (`DIARIZATION_CHUNK_DURATION`), it splits the (potentially preprocessed) audio into smaller overlapping chunks.
        *   Creates a temporary directory (e.g., `diarize_chunks_task_id_...`) within `settings.RESULTS_DIR`.
        *   Uses `ffmpeg` again to create multiple temporary audio chunk files (e.g., `audio_chunk_000.wav`) inside this directory.
    *   **Pipeline Execution:** Runs the `pyannote.audio` pipeline on the preprocessed file or individual chunks. This processing happens in memory.

4.  **Cleanup Logic (`diarize_file` finally block):**
    *   The main `diarize_file` function contains a `finally` block that executes regardless of success or failure within the function (lines 573-587).
    *   **Processed File Cleanup:** It checks if a temporary processed file was created (`is_temp_processed_file`) and explicitly removes it using `os.remove()`.
    *   **Chunk Directory Cleanup:** It checks if a temporary chunk directory (`temp_chunk_dir`) was created and explicitly removes the entire directory and its contents using `shutil.rmtree()`.

5.  **Output:** Returns a list of dictionaries, each representing a speaker segment with start time, end time, and speaker label. Does not persist this result itself.

## Privacy Considerations

*   **Intermediate File Creation:** The service *does* create intermediate temporary files (processed audio, audio chunks) on disk during its operation.
*   **Self-Contained Cleanup:** Crucially, the service implements its **own robust cleanup mechanism** within the `finally` block of its main execution function (`diarize_file`). This ensures that the temporary files it creates are reliably deleted upon completion or failure of the diarization step for a specific task.
*   **Cleanup Scope:** This cleanup is specific to the intermediate files created *by the diarization service*. It does not handle the original input file uploaded by the user (that's the `TaskManager`'s responsibility).
*   **Location:** Temporary files are created within the configured `RESULTS_DIR`, providing some isolation.

## Conclusion

The `DiarizationService` manages its own temporary intermediate files effectively. It creates necessary processed and chunked audio files but includes explicit and reliable cleanup logic within a `finally` block to remove them afterwards. This self-contained cleanup ensures that these intermediate files do not persist unnecessarily, complementing the `TaskManager`'s cleanup of the original input file. From a temporary file perspective, its privacy handling is sound.