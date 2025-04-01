# Privacy Review: app/services/processor.py

This file contains the core logic for coordinating transcription and optional diarization tasks.

## Analysis

1.  **Input:**
    *   Receives task parameters (`task_params`) including the `file_path` to the audio file (stored temporarily by the API layer).
    *   Receives `task_id` for logging and context.

2.  **Processing:**
    *   Loads the appropriate Whisper model via `ModelRegistry`. Models are potentially kept in memory for performance.
    *   Calls `run_transcription` which executes the model's `transcribe` method in a separate thread (`asyncio.to_thread`).
    *   Optionally calls `run_diarization` which uses the `DiarizationService`.
    *   Combines transcription and diarization results using `assign_speakers_to_segments`.
    *   All core processing (transcription, diarization result handling, segment assignment) happens primarily in memory using Python objects (dictionaries, lists, potentially Pandas DataFrames from diarization).

3.  **Output:**
    *   Returns a dictionary (`final_result`) containing the full transcription text, segments with timestamps (and potentially speaker labels), detected language, model used, and processing times.
    *   **Crucially, this module itself does not save the `final_result` to disk or any persistent storage.** It returns the result to the caller (the `TaskManager`).

4.  **Memory Management:**
    *   Includes explicit calls to `gc.collect()` and `torch.cuda.empty_cache()` at the end of `process_audio` (lines 452-454) and before potentially running diarization (lines 375-377) to help release GPU memory used by models.
    *   Relies on standard Python garbage collection for other objects.
    *   No secure memory wiping is performed (standard for Python).

5.  **File Handling:**
    *   Reads the input audio file specified by `file_path`.
    *   Uses `ffprobe` via `estimate_audio_duration` to read the file for duration estimation, but this doesn't create persistent intermediates.
    *   Does **not** delete the input `file_path`. Deletion responsibility lies elsewhere (handled by `TaskManager`).
    *   Does **not** create persistent output files containing the transcription result.

## Privacy Considerations

*   **In-Memory Processing:** Processing data primarily in memory is generally good for privacy as it minimizes persistent traces on disk *during this stage*.
*   **No Result Persistence:** The processor correctly avoids saving the final transcription result itself, leaving that responsibility to the caller.
*   **Input File Persistence:** The input file remains on disk after this processor finishes. Its deletion is critical and handled by the `TaskManager`.
*   **Model Persistence:** Models loaded in memory might contain weights, but this is standard practice. They are not user-specific data.
*   **Memory Remanence:** Standard GC and cache clearing do not guarantee secure wiping of data from physical memory. This is a general limitation, usually mitigated by OS memory management.

## Conclusion

`processor.py` handles the core computation. From a privacy perspective, its main strengths are processing in memory and not persisting the results itself. It relies correctly on the `TaskManager` for input file cleanup. It's well-suited to provide results to different handling mechanisms (like the current `TaskManager` storage or a future session-based storage).