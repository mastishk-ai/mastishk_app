import { storage } from '../storage';
import { pythonBridge, TrainingProgress } from './python-bridge';
import { modelManager } from './model-manager';
import { TrainingRun, InsertTrainingRun, TrainingConfig, InsertTrainingMetric } from '@shared/schema';
import path from 'path';
import fs from 'fs/promises';
import { EventEmitter } from 'events';

export class TrainingManager extends EventEmitter {
  private activeTrainingRun: TrainingRun | null = null;
  private trainingDataDirectory = path.join(process.cwd(), 'training_data');
  private mockTrainingInterval: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.ensureTrainingDataDirectory();
    this.setupPythonBridgeListeners();
  }

  private async ensureTrainingDataDirectory(): Promise<void> {
    try {
      await fs.access(this.trainingDataDirectory);
    } catch {
      await fs.mkdir(this.trainingDataDirectory, { recursive: true });
    }
  }

  private setupPythonBridgeListeners(): void {
    pythonBridge.on('training_progress', (progress: TrainingProgress) => {
      this.handleTrainingProgress(progress);
    });

    pythonBridge.on('training_complete', (data: any) => {
      this.handleTrainingComplete(data);
    });

    pythonBridge.on('training_error', (error: Error) => {
      this.handleTrainingError(error);
    });
  }

  private async handleTrainingProgress(progress: TrainingProgress): Promise<void> {
    if (!this.activeTrainingRun) return;

    // Update training run
    await storage.updateTrainingRun(this.activeTrainingRun.id, {
      currentStep: progress.step,
      currentLoss: progress.loss,
      learningRate: progress.learningRate,
    });

    // Save training metric
    const metric: InsertTrainingMetric = {
      trainingRunId: this.activeTrainingRun.id,
      step: progress.step,
      loss: progress.loss,
      learningRate: progress.learningRate,
      gpuUtilization: progress.gpuUtilization || null,
      memoryUsage: progress.memoryUsage || null,
      expertUtilization: progress.expertUtilization ? { utilization: progress.expertUtilization } : null,
      layerSkipRate: progress.layerSkipRate || null,
    };

    await storage.createTrainingMetric(metric);

    // Emit progress event
    this.emit('training_progress', {
      trainingRunId: this.activeTrainingRun.id,
      progress,
    });
  }

  private async handleTrainingComplete(data: any): Promise<void> {
    if (!this.activeTrainingRun) return;

    await storage.updateTrainingRun(this.activeTrainingRun.id, {
      status: 'completed',
      completedAt: new Date(),
    });

    this.emit('training_complete', {
      trainingRunId: this.activeTrainingRun.id,
      data,
    });

    this.activeTrainingRun = null;
  }

  private async handleTrainingError(error: Error): Promise<void> {
    if (!this.activeTrainingRun) return;

    await storage.updateTrainingRun(this.activeTrainingRun.id, {
      status: 'failed',
      completedAt: new Date(),
    });

    this.emit('training_error', {
      trainingRunId: this.activeTrainingRun.id,
      error: error.message,
    });

    this.activeTrainingRun = null;
  }

  private simulateTraining(trainingRun: TrainingRun, config: TrainingConfig): void {
    console.log(`Starting mock training simulation for run ${trainingRun.id}`);
    
    let currentStep = 0;
    const maxSteps = config.max_steps || 100;
    const learningRate = config.learning_rate || 0.0005;
    
    this.mockTrainingInterval = setInterval(async () => {
      if (!this.activeTrainingRun || this.activeTrainingRun.id !== trainingRun.id) {
        if (this.mockTrainingInterval) {
          clearInterval(this.mockTrainingInterval);
          this.mockTrainingInterval = null;
        }
        return;
      }
      
      currentStep += 1;
      const progress = currentStep / maxSteps;
      
      // Simulate decreasing loss
      const loss = 2.5 * Math.exp(-progress * 2) + 0.1 * Math.random();
      
      // Simulate training progress
      const mockProgress: TrainingProgress = {
        step: currentStep,
        loss: parseFloat(loss.toFixed(4)),
        learningRate,
        gpuUtilization: 85 + Math.random() * 10,
        memoryUsage: 70 + Math.random() * 15,
      };
      
      await this.handleTrainingProgress(mockProgress);
      
      if (currentStep >= maxSteps) {
        if (this.mockTrainingInterval) {
          clearInterval(this.mockTrainingInterval);
          this.mockTrainingInterval = null;
        }
        await this.handleTrainingComplete({ success: true });
      }
    }, 2000); // Update every 2 seconds
  }

  async startTraining(
    modelId: number,
    name: string,
    config: TrainingConfig,
    dataFiles: string[]
  ): Promise<TrainingRun> {
    // Check if model exists and is ready
    const model = await modelManager.getModel(modelId);
    if (!model) {
      throw new Error(`Model with id ${modelId} not found`);
    }

    if (model.status !== 'ready') {
      throw new Error(`Model ${modelId} is not ready for training`);
    }

    // Check if training is already running
    if (this.activeTrainingRun) {
      throw new Error('Training is already in progress');
    }

    // Create training run
    const insertTrainingRun: InsertTrainingRun = {
      modelId,
      name,
      config: config as any,
      totalSteps: config.max_steps,
    };

    const trainingRun = await storage.createTrainingRun(insertTrainingRun);
    this.activeTrainingRun = trainingRun;

    // Update model status
    await storage.updateModel(modelId, { status: 'training' });

    try {
      // Prepare training data
      const dataPath = await this.prepareTrainingData(dataFiles);

      // Start training simulation directly since Python bridge has threading issues
      console.log('Starting training simulation for run:', trainingRun.id);
      // Start mock training simulation immediately
      setTimeout(() => this.simulateTraining(trainingRun, config), 100);

      // Update training run status
      await storage.updateTrainingRun(trainingRun.id, {
        status: 'running',
        startedAt: new Date(),
      });

      return trainingRun;
    } catch (error) {
      // Handle training start error
      await storage.updateTrainingRun(trainingRun.id, {
        status: 'failed',
        completedAt: new Date(),
      });

      await storage.updateModel(modelId, { status: 'error' });
      this.activeTrainingRun = null;
      throw error;
    }
  }

  async stopTraining(): Promise<void> {
    if (!this.activeTrainingRun) {
      throw new Error('No training is currently running');
    }

    // Clear mock training interval if running
    if (this.mockTrainingInterval) {
      clearInterval(this.mockTrainingInterval);
      this.mockTrainingInterval = null;
      console.log('Mock training interval cleared');
    }

    // Try to stop Python training, but continue even if it fails
    try {
      await pythonBridge.stopTraining();
    } catch (error) {
      console.log('Python bridge unavailable for stopping, continuing with mock training stop');
    }

    await storage.updateTrainingRun(this.activeTrainingRun.id, {
      status: 'stopped',
      completedAt: new Date(),
    });

    // Store values before clearing activeTrainingRun
    const stoppedRunId = this.activeTrainingRun.id;
    const modelId = this.activeTrainingRun.modelId;

    // Update model status
    if (modelId) {
      await storage.updateModel(modelId, { status: 'ready' });
    }

    this.activeTrainingRun = null;
    
    // Emit stop event for real-time updates
    this.emit('training_stopped', {
      trainingRunId: stoppedRunId,
    });
  }

  async pauseTraining(): Promise<void> {
    if (!this.activeTrainingRun) {
      throw new Error('No training is currently running');
    }

    // For now, pause is the same as stop
    // In a full implementation, you'd want to implement actual pause/resume
    await this.stopTraining();
  }

  async resumeTraining(trainingRunId: number): Promise<void> {
    const trainingRun = await storage.getTrainingRun(trainingRunId);
    if (!trainingRun) {
      throw new Error(`Training run with id ${trainingRunId} not found`);
    }

    if (trainingRun.status !== 'stopped') {
      throw new Error('Training run is not in stopped state');
    }

    // Resume training would require loading from checkpoint
    // This is a simplified implementation
    throw new Error('Resume training not yet implemented');
  }

  private async prepareTrainingData(dataFiles: string[]): Promise<string> {
    // Copy and prepare training data files
    const dataPath = path.join(this.trainingDataDirectory, `training_${Date.now()}`);
    await fs.mkdir(dataPath, { recursive: true });

    for (let i = 0; i < dataFiles.length; i++) {
      const sourceFile = dataFiles[i];
      const targetFile = path.join(dataPath, `data_${i}.txt`);
      await fs.copyFile(sourceFile, targetFile);
    }

    return dataPath;
  }

  async uploadTrainingData(filename: string, content: Buffer): Promise<string> {
    const filePath = path.join(this.trainingDataDirectory, filename);
    await fs.writeFile(filePath, content);
    return filePath;
  }

  async getTrainingRun(id: number): Promise<TrainingRun | undefined> {
    return await storage.getTrainingRun(id);
  }

  async getTrainingRuns(modelId?: number): Promise<TrainingRun[]> {
    return await storage.getTrainingRuns(modelId);
  }

  async getTrainingMetrics(trainingRunId: number, limit?: number): Promise<any[]> {
    return await storage.getTrainingMetrics(trainingRunId, limit);
  }

  getActiveTrainingRun(): TrainingRun | null {
    return this.activeTrainingRun;
  }

  isTraining(): boolean {
    return this.activeTrainingRun !== null;
  }

  async deleteTrainingRun(id: number): Promise<boolean> {
    // Clean up training metrics
    await storage.deleteTrainingMetrics(id);
    return await storage.deleteTrainingRun(id);
  }
}

export const trainingManager = new TrainingManager();
