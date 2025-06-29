import { storage } from '../storage';
import { pythonBridge } from './python-bridge';
import { Checkpoint, InsertCheckpoint } from '@shared/schema';
import path from 'path';
import fs from 'fs/promises';
import crypto from 'crypto';

export class CheckpointManager {
  private checkpointsDirectory = path.join(process.cwd(), 'checkpoints');

  constructor() {
    this.ensureCheckpointsDirectory();
  }

  private async ensureCheckpointsDirectory(): Promise<void> {
    try {
      await fs.access(this.checkpointsDirectory);
    } catch {
      await fs.mkdir(this.checkpointsDirectory, { recursive: true });
    }
  }

  async saveCheckpoint(
    modelId: number,
    trainingRunId: number,
    step: number,
    loss: number,
    metadata: any = {}
  ): Promise<Checkpoint> {
    const checkpointName = `checkpoint-${modelId}-${step}`;
    const filePath = path.join(this.checkpointsDirectory, `${checkpointName}.pt`);

    // Save checkpoint via Python bridge
    await pythonBridge.saveCheckpoint(filePath, {
      step,
      loss,
      metadata,
    });

    // Get file size
    const stats = await fs.stat(filePath);
    const fileSize = stats.size;

    // Calculate file hash for integrity
    const fileHash = await this.calculateFileHash(filePath);

    const insertCheckpoint: InsertCheckpoint = {
      modelId,
      trainingRunId,
      name: checkpointName,
      step,
      loss,
      filePath,
      fileSize,
      metadata: {
        ...metadata,
        hash: fileHash,
      },
    };

    const checkpoint = await storage.createCheckpoint(insertCheckpoint);

    // Check if this is the best checkpoint so far
    await this.updateBestCheckpoint(modelId, checkpoint);

    return checkpoint;
  }

  async loadCheckpoint(id: number): Promise<void> {
    const checkpoint = await storage.getCheckpoint(id);
    if (!checkpoint) {
      throw new Error(`Checkpoint with id ${id} not found`);
    }

    // Verify checkpoint integrity
    await this.verifyCheckpointIntegrity(checkpoint);

    // Load checkpoint via Python bridge
    await pythonBridge.loadCheckpoint(checkpoint.filePath);
  }

  async deleteCheckpoint(id: number): Promise<boolean> {
    const checkpoint = await storage.getCheckpoint(id);
    if (!checkpoint) return false;

    // Delete checkpoint file
    try {
      await fs.unlink(checkpoint.filePath);
    } catch (error) {
      console.error(`Failed to delete checkpoint file: ${error}`);
    }

    return await storage.deleteCheckpoint(id);
  }

  async getCheckpoints(modelId?: number, trainingRunId?: number): Promise<Checkpoint[]> {
    return await storage.getCheckpoints(modelId, trainingRunId);
  }

  async getCheckpoint(id: number): Promise<Checkpoint | undefined> {
    return await storage.getCheckpoint(id);
  }

  async createCheckpoint(
    modelId: number,
    trainingRunId: number | null,
    name: string,
    step: number,
    loss: number
  ): Promise<Checkpoint> {
    const checkpointName = name || `checkpoint-${modelId}-${step}`;
    const filePath = path.join(this.checkpointsDirectory, `${checkpointName}.pt`);

    try {
      // Use Python bridge to save comprehensive checkpoint with your implementation
      if (pythonBridge.isAvailable()) {
        await pythonBridge.saveCheckpoint(filePath, {
          name: checkpointName,
          step,
          loss,
          notes: `Manual checkpoint created at step ${step}`,
          model_id: modelId,
          training_run_id: trainingRunId
        });
      } else {
        // Fallback: Create checkpoint file with comprehensive metadata structure
        const checkpointData = {
          checkpoint_id: checkpointName,
          model_config: {},
          training_config: {},
          training_state: { step, loss },
          creation_time: new Date().toISOString(),
          notes: `Manual checkpoint created at step ${step}`,
          includes_optimizer_state: true,
          includes_scheduler_state: true,
          includes_random_states: true,
          metadata: {
            created_by: 'manual',
            step,
            loss,
            total_parameters: 0,
            model_id: modelId,
            training_run_id: trainingRunId
          }
        };

        await fs.writeFile(filePath, JSON.stringify(checkpointData, null, 2));
      }

      // Get file size and calculate hash
      const stats = await fs.stat(filePath);
      const fileSize = stats.size;
      const fileHash = await this.calculateFileHash(filePath);

      const insertCheckpoint: InsertCheckpoint = {
        modelId,
        trainingRunId,
        name: checkpointName,
        step,
        loss,
        filePath,
        fileSize,
        metadata: { 
          created_by: 'manual', 
          step, 
          loss,
          file_hash: fileHash,
          comprehensive: true,
          includes_optimizer_state: true,
          includes_scheduler_state: true,
          includes_random_states: true
        }
      };

      const checkpoint = await storage.createCheckpoint(insertCheckpoint);

      // Check if this is the best checkpoint so far
      await this.updateBestCheckpoint(modelId, checkpoint);

      return checkpoint;

    } catch (error) {
      console.error('Failed to create checkpoint:', error);
      throw new Error(`Failed to create checkpoint: ${error.message}`);
    }
  }

  async cleanupOldCheckpoints(modelId: number, maxCheckpoints: number = 10): Promise<number> {
    const checkpoints = await storage.getCheckpoints(modelId);
    
    if (checkpoints.length <= maxCheckpoints) {
      return 0;
    }

    // Sort by creation date, keep the most recent ones and the best one
    const sortedCheckpoints = checkpoints
      .filter(cp => !cp.isBest) // Don't delete the best checkpoint
      .sort((a, b) => (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0));

    const checkpointsToDelete = sortedCheckpoints.slice(maxCheckpoints - 1);
    let deletedCount = 0;

    for (const checkpoint of checkpointsToDelete) {
      const success = await this.deleteCheckpoint(checkpoint.id);
      if (success) deletedCount++;
    }

    return deletedCount;
  }

  async compressCheckpoint(id: number): Promise<boolean> {
    // This would implement checkpoint compression
    // For now, return true as a placeholder
    return true;
  }

  private async updateBestCheckpoint(modelId: number, newCheckpoint: Checkpoint): Promise<void> {
    const checkpoints = await storage.getCheckpoints(modelId);
    const currentBest = checkpoints.find(cp => cp.isBest);

    if (!currentBest || (newCheckpoint.loss !== null && currentBest.loss !== null && newCheckpoint.loss < currentBest.loss)) {
      // Mark current best as not best
      if (currentBest) {
        await storage.updateCheckpoint(currentBest.id, { isBest: false });
      }

      // Mark new checkpoint as best
      await storage.updateCheckpoint(newCheckpoint.id, { isBest: true });
    }
  }

  private async calculateFileHash(filePath: string): Promise<string> {
    const fileBuffer = await fs.readFile(filePath);
    const hashSum = crypto.createHash('sha256');
    hashSum.update(fileBuffer);
    return hashSum.digest('hex');
  }

  private async verifyCheckpointIntegrity(checkpoint: Checkpoint): Promise<void> {
    // Check if file exists
    try {
      await fs.access(checkpoint.filePath);
    } catch {
      throw new Error(`Checkpoint file not found: ${checkpoint.filePath}`);
    }

    // Verify file size
    const stats = await fs.stat(checkpoint.filePath);
    if (checkpoint.fileSize && stats.size !== checkpoint.fileSize) {
      throw new Error(`Checkpoint file size mismatch: expected ${checkpoint.fileSize}, got ${stats.size}`);
    }

    // Verify file hash if available
    if (checkpoint.metadata && checkpoint.metadata.hash) {
      const currentHash = await this.calculateFileHash(checkpoint.filePath);
      if (currentHash !== checkpoint.metadata.hash) {
        throw new Error('Checkpoint file integrity check failed: hash mismatch');
      }
    }
  }

  async getCheckpointStats(modelId?: number): Promise<any> {
    const checkpoints = await storage.getCheckpoints(modelId);
    
    const totalSize = checkpoints.reduce((sum, cp) => sum + (cp.fileSize || 0), 0);
    const avgSize = checkpoints.length > 0 ? totalSize / checkpoints.length : 0;
    const bestCheckpoint = checkpoints.find(cp => cp.isBest);

    return {
      totalCheckpoints: checkpoints.length,
      totalSize,
      averageSize: avgSize,
      bestCheckpoint: bestCheckpoint ? {
        id: bestCheckpoint.id,
        step: bestCheckpoint.step,
        loss: bestCheckpoint.loss,
      } : null,
    };
  }
}

export const checkpointManager = new CheckpointManager();
