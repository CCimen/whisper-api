import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

// Configurable API endpoint
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Transcriber component for audio file transcription with diarization
 * 
 * Features:
 * - File upload with drag & drop
 * - Progress tracking
 * - Model selection
 * - Diarization toggle
 * - Results display with speakers
 * - Status updates
 */
const Transcriber = () => {
  // State for the component
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState('medium');
  const [enableDiarization, setEnableDiarization] = useState(false);
  
  // References
  const pollInterval = useRef(null);
  const fileInputRef = useRef(null);
  
  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };
  
  // Handle drag and drop
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      
      // Validate file type
      const validTypes = ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/ogg', 'audio/flac'];
      if (validTypes.includes(droppedFile.type) || 
          droppedFile.name.match(/\.(mp3|wav|ogg|flac|m4a)$/)) {
        setFile(droppedFile);
        setError(null);
      } else {
        setError('Please upload an audio file (MP3, WAV, OGG, FLAC)');
      }
    }
  };
  
  // Handle drag events
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  // Start transcription
  const startTranscription = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }
    
    try {
      setIsUploading(true);
      setError(null);
      
      // Create form data
      const formData = new FormData();
      formData.append('audio_file', file);
      formData.append('model_size', selectedModel);
      formData.append('diarization', enableDiarization);
      
      // Submit file for transcription
      const response = await axios.post(`${API_URL}/transcriptions`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      const { id, status: initialStatus } = response.data;
      setJobId(id);
      setStatus(initialStatus);
      setIsUploading(false);
      setIsProcessing(true);
      
      // Start polling for status
      startPolling(id);
    } catch (error) {
      setIsUploading(false);
      setError(error.response?.data?.detail || 'Error uploading file');
      console.error('Upload error:', error);
    }
  };
  
  // Start polling for status updates
  const startPolling = (id) => {
    // Clear any existing interval
    if (pollInterval.current) {
      clearInterval(pollInterval.current);
    }
    
    // Set up polling
    pollInterval.current = setInterval(async () => {
      try {
        const response = await axios.get(`${API_URL}/transcriptions/${id}/status`);
        const { status, progress, error } = response.data;
        
        setStatus(status);
        setProgress(progress * 100);
        
        if (error) {
          setError(error);
          clearInterval(pollInterval.current);
          setIsProcessing(false);
        }
        
        if (status === 'completed') {
          clearInterval(pollInterval.current);
          fetchResults(id);
        }
      } catch (error) {
        console.error('Polling error:', error);
        setError('Error checking transcription status');
        clearInterval(pollInterval.current);
        setIsProcessing(false);
      }
    }, 1000);
  };
  
  // Fetch final results
  const fetchResults = async (id) => {
    try {
      const response = await axios.get(`${API_URL}/transcriptions/${id}`);
      setResult(response.data);
      setIsProcessing(false);
    } catch (error) {
      console.error('Result fetch error:', error);
      setError('Error fetching transcription results');
      setIsProcessing(false);
    }
  };
  
  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollInterval.current) {
        clearInterval(pollInterval.current);
      }
    };
  }, []);
  
  // Reset the form
  const handleReset = () => {
    setFile(null);
    setIsProcessing(false);
    setJobId(null);
    setStatus('idle');
    setProgress(0);
    setResult(null);
    setError(null);
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    
    if (pollInterval.current) {
      clearInterval(pollInterval.current);
      pollInterval.current = null;
    }
  };
  
  // Format the timestamp
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };
  
  // Determine status message
  const getStatusMessage = () => {
    switch (status) {
      case 'idle':
        return 'Ready to transcribe';
      case 'pending':
        return 'Preparing transcription...';
      case 'transcribing':
        return 'Transcribing audio...';
      case 'diarizing':
        return 'Identifying speakers...';
      case 'completing':
        return 'Finalizing results...';
      case 'completed':
        return 'Transcription complete!';
      case 'error':
        return 'Error occurred';
      default:
        return 'Processing...';
    }
  };
  
  // Get color for speaker
  const getSpeakerColor = (speaker) => {
    const colors = {
      'SPEAKER_0': '#4299e1', // blue
      'SPEAKER_1': '#48bb78', // green
      'SPEAKER_2': '#ed8936', // orange
      'SPEAKER_3': '#9f7aea', // purple
      'SPEAKER_4': '#f56565', // red
      'UNKNOWN': '#a0aec0'   // gray
    };
    
    return colors[speaker] || '#a0aec0';
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">Audio Transcription</h1>
      
      {!isProcessing && !result && (
        <div 
          className="border-2 border-dashed rounded-lg p-8 mb-6 text-center"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
            accept="audio/*"
          />
          
          {file ? (
            <div>
              <p className="mb-2">Selected file: <span className="font-semibold">{file.name}</span></p>
              <p className="text-sm text-gray-600">
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          ) : (
            <div>
              <p className="mb-2">Drag and drop your audio file here, or</p>
              <button 
                onClick={() => fileInputRef.current.click()}
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
              >
                Select File
              </button>
            </div>
          )}
        </div>
      )}
      
      {!isProcessing && !result && (
        <div className="mb-6">
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Model Size:</label>
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full p-2 border rounded"
            >
              <option value="tiny">Tiny (Fastest, Less Accurate)</option>
              <option value="small">Small (Fast, Good Accuracy)</option>
              <option value="medium">Medium (Balanced)</option>
              <option value="large">Large (Slow, Most Accurate)</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Larger models are more accurate but slower to process.
            </p>
          </div>
          
          <div className="mb-4">
            <label className="flex items-center">
              <input 
                type="checkbox" 
                checked={enableDiarization} 
                onChange={(e) => setEnableDiarization(e.target.checked)}
                className="mr-2"
              />
              <span>Enable Speaker Identification (Diarization)</span>
            </label>
            <p className="text-xs text-gray-500 ml-5">
              Identifies different speakers in the audio. Adds processing time.
            </p>
          </div>
          
          <button 
            onClick={startTranscription}
            disabled={!file || isUploading}
            className={`w-full py-2 px-4 rounded font-semibold ${
              !file || isUploading 
                ? 'bg-gray-300 cursor-not-allowed' 
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            {isUploading ? 'Uploading...' : 'Start Transcription'}
          </button>
        </div>
      )}
      
      {isProcessing && (
        <div className="mb-6">
          <h2 className="text-xl font-semibold mb-2">{getStatusMessage()}</h2>
          
          <div className="w-full bg-gray-200 rounded-full h-4 mb-2">
            <div 
              className="bg-blue-500 h-4 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          
          <p className="text-sm text-gray-600">
            {Math.round(progress)}% complete
          </p>
          
          <button 
            onClick={handleReset}
            className="mt-4 text-red-500 hover:text-red-700"
          >
            Cancel
          </button>
        </div>
      )}
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      )}
      
      {result && (
        <div>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold">Transcription Results</h2>
            <button 
              onClick={handleReset}
              className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
            >
              New Transcription
            </button>
          </div>
          
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2">Summary</h3>
            <ul className="bg-gray-50 rounded p-4">
              <li><strong>Duration:</strong> {formatTime(result.duration)} ({result.duration.toFixed(2)} seconds)</li>
              <li><strong>Processing Time:</strong> {result.processing_time.toFixed(2)} seconds</li>
              <li><strong>Realtime Factor:</strong> {(result.duration / result.processing_time).toFixed(2)}x</li>
              {result.speakers && result.speakers.length > 0 && (
                <li><strong>Speakers Identified:</strong> {result.speakers.length}</li>
              )}
            </ul>
          </div>
          
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2">Full Transcription</h3>
            <div className="bg-white border rounded p-4 whitespace-pre-wrap">
              {result.transcription}
            </div>
          </div>
          
          {result.segments && result.segments.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold mb-2">Segments</h3>
              <div className="border rounded overflow-hidden">
                {result.segments.map((segment, index) => (
                  <div 
                    key={index} 
                    className={`p-3 border-b ${index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}`}
                  >
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-600">
                        {formatTime(segment.start)} - {formatTime(segment.end)}
                      </span>
                      
                      {segment.speaker && (
                        <span 
                          className="text-sm font-semibold px-2 py-0.5 rounded-full"
                          style={{
                            backgroundColor: getSpeakerColor(segment.speaker) + '20',
                            color: getSpeakerColor(segment.speaker)
                          }}
                        >
                          {segment.speaker}
                        </span>
                      )}
                    </div>
                    <p>{segment.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Transcriber;