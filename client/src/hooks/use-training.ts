import { useState, useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '@/lib/queryClient';
import { TrainingConfig } from '@shared/schema';
import { TrainingProgress } from '@/lib/types';

export function useTraining() {
  const [config, setConfig] = useState<TrainingConfig>({
    learning_rate: 5e-4,
    batch_size: 2,
    gradient_accumulation_steps: 4,
    max_steps: 10000,
    eval_steps: 500,
    save_steps: 1000,
    warmup_steps: 100,
    max_grad_norm: 1.0,
    weight_decay: 0.01,
    mixed_precision: true,
    gradient_checkpointing: true,
    early_stopping: false,
    early_stopping_patience: 5,
    early_stopping_threshold: 1e-4,
    use_wandb: false,
    use_tensorboard: false,
    seed: 42,
    save_optimizer_state: true,
    save_scheduler_state: true,
    save_random_states: true,
    verify_integrity: true,
    max_checkpoints: 10,
  });

  const queryClient = useQueryClient();

  const updateConfig = useCallback((updates: Partial<TrainingConfig>) => {
    setConfig(prev => ({ ...prev, ...updates }));
  }, []);

  // Training status query
  const { data: trainingStatus } = useQuery({
    queryKey: ['/api/training/status'],
    refetchInterval: 2000, // Poll every 2 seconds
  });

  // Training runs query
  const { data: trainingRuns } = useQuery({
    queryKey: ['/api/training-runs'],
  });

  // Start training mutation
  const startTrainingMutation = useMutation({
    mutationFn: async ({ modelId, name, dataFiles }: { 
      modelId: number; 
      name: string; 
      dataFiles: string[];
    }) => {
      const response = await apiRequest('POST', '/api/training-runs', {
        modelId,
        name,
        config,
        dataFiles,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/training-runs'] });
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
    },
  });

  // Stop training mutation
  const stopTrainingMutation = useMutation({
    mutationFn: async (runId: number) => {
      const response = await apiRequest('POST', `/api/training-runs/${runId}/stop`);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
      queryClient.invalidateQueries({ queryKey: ['/api/training-runs'] });
    },
  });

  // Pause training mutation
  const pauseTrainingMutation = useMutation({
    mutationFn: async (runId: number) => {
      const response = await apiRequest('POST', `/api/training-runs/${runId}/pause`);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
    },
  });

  // Resume training mutation
  const resumeTrainingMutation = useMutation({
    mutationFn: async (runId: number) => {
      const response = await apiRequest('POST', `/api/training-runs/${runId}/resume`);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
    },
  });

  // File upload mutation
  const uploadDataMutation = useMutation({
    mutationFn: async (files: FileList) => {
      const formData = new FormData();
      Array.from(files).forEach(file => {
        formData.append('files', file);
      });

      const response = await fetch('/api/upload/training-data', {
        method: 'POST',
        body: formData,
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      return response.json();
    },
    onSuccess: () => {
      // Invalidate any data-related queries
      queryClient.invalidateQueries({ queryKey: ['/api/training-data'] });
    },
  });

  const getTrainingMetrics = useCallback(async (runId: number, limit?: number) => {
    const response = await apiRequest('GET', `/api/training-runs/${runId}/metrics${limit ? `?limit=${limit}` : ''}`);
    return response.json();
  }, []);

  const isTraining = (trainingStatus as any)?.isTraining || false;
  const activeRun = (trainingStatus as any)?.activeRun;

  return {
    config,
    updateConfig,
    trainingRuns: trainingRuns || [],
    trainingStatus: trainingStatus || {},
    isTraining,
    activeRun,
    startTraining: startTrainingMutation.mutate,
    stopTraining: stopTrainingMutation.mutate,
    pauseTraining: pauseTrainingMutation.mutate,
    resumeTraining: resumeTrainingMutation.mutate,
    uploadData: uploadDataMutation.mutateAsync,
    uploadDataMutation,
    getTrainingMetrics,
    isStarting: startTrainingMutation.isPending,
    isStopping: stopTrainingMutation.isPending,
    isPausing: pauseTrainingMutation.isPending,
    isResuming: resumeTrainingMutation.isPending,
    isUploading: uploadDataMutation.isPending,
  };
}
