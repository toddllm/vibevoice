#!/usr/bin/env python3
"""
Batch Processing Pipeline for VibeVoice
Handles async text-to-audio conversion with job queue
"""

import uuid
import time
import json
import os
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING_TEXT = "processing_text"
    PROCESSING_AUDIO = "processing_audio"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BatchJob:
    """Batch processing job"""
    job_id: str
    original_text: str
    status: JobStatus
    created_at: float
    updated_at: float
    
    # Results
    normalized_text: Optional[str] = None
    voice_assignments: Optional[Dict[int, str]] = None
    audio_file: Optional[str] = None
    audio_duration: Optional[float] = None
    
    # Metadata
    processing_time_text: Optional[float] = None
    processing_time_audio: Optional[float] = None
    error_message: Optional[str] = None

class BatchProcessor:
    """Handles async batch processing of text-to-audio conversion"""
    
    def __init__(self, llm_processor, vibevoice_server, results_dir: str = "batch_results"):
        self.llm_processor = llm_processor
        self.vibevoice_server = vibevoice_server
        self.results_dir = results_dir
        self.jobs: Dict[str, BatchJob] = {}
        self.processing_thread = None
        self.running = True
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Start background processing thread
        self._start_processing_thread()
        
        logger.info("BatchProcessor initialized")
    
    def submit_job(self, text: str) -> str:
        """Submit a new batch job and return job ID"""
        job_id = str(uuid.uuid4())[:8]  # Short ID for user convenience
        
        job = BatchJob(
            job_id=job_id,
            original_text=text,
            status=JobStatus.QUEUED,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.jobs[job_id] = job
        logger.info(f"Job {job_id} submitted: {len(text)} characters")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current status of a job"""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        return {
            "job_id": job_id,
            "status": job.status.value,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "original_text_preview": job.original_text[:200] + "..." if len(job.original_text) > 200 else job.original_text,
            "normalized_text": job.normalized_text,
            "voice_assignments": job.voice_assignments,
            "audio_file": job.audio_file,
            "audio_duration": job.audio_duration,
            "processing_time_text": job.processing_time_text,
            "processing_time_audio": job.processing_time_audio,
            "error_message": job.error_message
        }
    
    def list_jobs(self) -> List[Dict]:
        """List all jobs with their status"""
        return [
            {
                "job_id": job_id,
                "status": job.status.value,
                "created_at": job.created_at,
                "text_preview": job.original_text[:100] + "..." if len(job.original_text) > 100 else job.original_text,
                "completed": job.status == JobStatus.COMPLETED,
                "error": job.error_message
            }
            for job_id, job in sorted(self.jobs.items(), key=lambda x: x[1].created_at, reverse=True)
        ]
    
    def _start_processing_thread(self):
        """Start background processing thread"""
        def process_loop():
            while self.running:
                try:
                    # Find next queued job
                    queued_jobs = [(job_id, job) for job_id, job in self.jobs.items() 
                                 if job.status == JobStatus.QUEUED]
                    
                    if queued_jobs:
                        job_id, job = min(queued_jobs, key=lambda x: x[1].created_at)
                        self._process_job(job_id, job)
                    else:
                        time.sleep(1)  # Wait for new jobs
                        
                except Exception as e:
                    logger.error(f"Processing thread error: {e}")
                    time.sleep(5)
        
        self.processing_thread = threading.Thread(target=process_loop, daemon=True)
        self.processing_thread.start()
    
    def _process_job(self, job_id: str, job: BatchJob):
        """Process a single batch job with model staging"""
        try:
            logger.info(f"Starting batch job {job_id} with staging")
            
            # Step 1: Text processing with LLM (qwen3:8b)
            job.status = JobStatus.PROCESSING_TEXT
            job.updated_at = time.time()
            
            start_time = time.time()
            normalized_text, synthesis_units = self.llm_processor.process(job.original_text, force_llm=True)
            job.processing_time_text = time.time() - start_time
            
            job.normalized_text = normalized_text
            job.voice_assignments = getattr(self.llm_processor, '_last_voice_assignments', {})
            job.updated_at = time.time()
            
            logger.info(f"Job {job_id} text processing completed in {job.processing_time_text:.1f}s")
            logger.info(f"Job {job_id} normalized to {len(synthesis_units)} synthesis units")
            
            # Step 1.5: Free GPU memory by unloading LLM model
            logger.info(f"Job {job_id} unloading LLM model to free GPU memory")
            self._unload_llm_model()
            
            # Step 2: Audio synthesis (after freeing GPU memory)
            job.status = JobStatus.PROCESSING_AUDIO
            job.updated_at = time.time()
            
            start_time = time.time()
            audio_result = self._synthesize_audio(normalized_text)
            job.processing_time_audio = time.time() - start_time
            
            # Save audio file
            audio_filename = f"{job_id}_audio.wav"
            audio_path = os.path.join(self.results_dir, audio_filename)
            
            with open(audio_path, 'wb') as f:
                f.write(audio_result['audio_bytes'])
            
            job.audio_file = audio_filename
            job.audio_duration = audio_result.get('duration', 0)
            job.status = JobStatus.COMPLETED
            job.updated_at = time.time()
            
            # Save job metadata
            job_file = os.path.join(self.results_dir, f"{job_id}_job.json")
            with open(job_file, 'w') as f:
                job_dict = asdict(job)
                job_dict['status'] = job.status.value  # Convert enum to string
                json.dump(job_dict, f, indent=2)
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.updated_at = time.time()
            logger.error(f"Job {job_id} failed: {e}")
    
    def _unload_llm_model(self):
        """Unload LLM model from GPU to free memory"""
        try:
            import requests
            # Force garbage collection first
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            
            # Unload ollama model
            response = requests.post(
                'http://127.0.0.1:11434/api/generate',
                json={'model': 'qwen3:8b', 'keep_alive': 0},
                timeout=30
            )
            
            logger.info("LLM model unloaded from GPU memory")
            
        except Exception as e:
            logger.warning(f"Failed to unload LLM model: {e}")
    
    def _synthesize_audio(self, text: str) -> Dict:
        """Synthesize audio using existing synthesis endpoint"""
        try:
            import requests
            import base64
            
            # Use the working /synthesize endpoint 
            response = requests.post(
                'http://localhost:5000/synthesize',
                json={
                    'text': text,
                    'model': '7B',
                    'voice': 'demo/voices/en-Alice_woman.wav',
                    'format': 'base64'
                },
                timeout=600  # 10 minutes for synthesis
            )
            
            if response.status_code != 200:
                raise Exception(f"Synthesis endpoint failed: {response.text}")
            
            result = response.json()
            if 'error' in result:
                raise Exception(f"Synthesis error: {result['error']}")
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(result['audio'])
            
            return {
                'audio_bytes': audio_bytes,
                'duration': result.get('duration', 0),
                'rtf': result.get('rtf', 1.0)
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise Exception(f"Audio synthesis failed: {str(e)}")

# Global batch processor instance
batch_processor = None

def get_batch_processor():
    """Get or create batch processor instance"""
    global batch_processor
    if batch_processor is None:
        # Initialize with current LLM processor and server
        # This will be set when the server starts
        pass
    return batch_processor

def init_batch_processor(llm_processor, vibevoice_server):
    """Initialize the batch processor"""
    global batch_processor
    batch_processor = BatchProcessor(llm_processor, vibevoice_server)
    return batch_processor