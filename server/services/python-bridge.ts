import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import path from 'path';
import fs from 'fs/promises';
import { EventEmitter } from 'events';

export interface PythonMessage {
  type: string;
  data: any;
  timestamp: number;
}

export interface TrainingProgress {
  step: number;
  loss: number;
  learningRate: number;
  gpuUtilization?: number;
  memoryUsage?: number;
  expertUtilization?: number[];
  layerSkipRate?: number;
}

export interface GenerationResult {
  output: string;
  tokensGenerated: number;
  generationTime: number;
}

export class PythonBridge extends EventEmitter {
  private process: ChildProcessWithoutNullStreams | null = null;
  private isInitialized = false;
  private messageQueue: string[] = [];

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    const pythonPath = path.join(process.cwd(), 'python');
    const scriptPath = path.join(pythonPath, 'mastishk_bridge.py');

    try {
      // Verify Python script exists
      await fs.access(scriptPath);
    } catch (error) {
      throw new Error(`Python bridge script not found at ${scriptPath}`);
    }

    this.process = spawn('python3', [scriptPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: pythonPath,
    });

    this.process.stdout.on('data', (data) => {
      const messages = data.toString().trim().split('\n');
      for (const message of messages) {
        if (message.trim()) {
          try {
            const parsed = JSON.parse(message) as PythonMessage;
            this.emit('message', parsed);
            this.handleMessage(parsed);
          } catch (error) {
            console.error('Failed to parse Python message:', message);
          }
        }
      }
    });

    this.process.stderr.on('data', (data) => {
      console.error('Python bridge error:', data.toString());
      this.emit('error', new Error(data.toString()));
    });

    this.process.on('close', (code) => {
      console.log(`Python bridge process closed with code ${code}`);
      this.isInitialized = false;
      this.process = null;
    });

    // Wait for initialization
    await this.waitForMessage('initialized');
    this.isInitialized = true;
  }

  private handleMessage(message: PythonMessage): void {
    switch (message.type) {
      case 'training_progress':
        this.emit('training_progress', message.data as TrainingProgress);
        break;
      case 'training_complete':
        this.emit('training_complete', message.data);
        break;
      case 'training_error':
        this.emit('training_error', new Error(message.data.error));
        break;
      case 'generation_complete':
        this.emit('generation_complete', message.data as GenerationResult);
        break;
      case 'generation_error':
        this.emit('generation_error', new Error(message.data.error));
        break;
      case 'model_loaded':
        this.emit('model_loaded', message.data);
        break;
      case 'model_error':
        this.emit('model_error', new Error(message.data.error));
        break;
    }
  }

  private async waitForMessage(type: string, timeout = 30000): Promise<any> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Timeout waiting for message type: ${type}`));
      }, timeout);

      const handler = (message: PythonMessage) => {
        if (message.type === type) {
          clearTimeout(timer);
          this.off('message', handler);
          resolve(message.data);
        }
      };

      this.on('message', handler);
    });
  }

  private sendMessage(message: any): void {
    if (!this.process || !this.isInitialized) {
      throw new Error('Python bridge not initialized');
    }

    const messageStr = JSON.stringify(message) + '\n';
    this.process.stdin.write(messageStr);
  }

  async initializeModel(config: any): Promise<void> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    this.sendMessage({
      type: 'initialize_model',
      data: config,
    });

    await this.waitForMessage('model_loaded');
  }

  async startTraining(config: any, dataPath: string): Promise<void> {
    this.sendMessage({
      type: 'start_training',
      data: {
        config,
        dataPath,
      },
    });
  }

  async stopTraining(): Promise<void> {
    this.sendMessage({
      type: 'stop_training',
      data: {},
    });
  }

  async generateText(prompt: string, config: any): Promise<GenerationResult> {
    this.sendMessage({
      type: 'generate_text',
      data: {
        prompt,
        config,
      },
    });

    return await this.waitForMessage('generation_complete');
  }

  async saveCheckpoint(path: string, metadata: any): Promise<void> {
    this.sendMessage({
      type: 'save_checkpoint',
      data: {
        path,
        metadata,
      },
    });

    await this.waitForMessage('checkpoint_saved');
  }

  async loadCheckpoint(path: string): Promise<void> {
    this.sendMessage({
      type: 'load_checkpoint',
      data: {
        path,
      },
    });

    await this.waitForMessage('checkpoint_loaded');
  }

  isAvailable(): boolean {
    return this.isInitialized && this.process !== null;
  }

  async cleanup(): Promise<void> {
    if (this.process) {
      this.sendMessage({
        type: 'cleanup',
        data: {},
      });

      this.process.kill();
      this.process = null;
      this.isInitialized = false;
    }
  }
}

export const pythonBridge = new PythonBridge();
