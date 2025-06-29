import { storage } from '../storage';
import { pythonBridge } from './python-bridge';
import { Model, InsertModel, ModelConfig } from '@shared/schema';
import path from 'path';
import fs from 'fs/promises';

export class ModelManager {
  private activeModel: Model | null = null;
  private modelsDirectory = path.join(process.cwd(), 'models');

  constructor() {
    this.ensureModelsDirectory();
  }

  private async ensureModelsDirectory(): Promise<void> {
    try {
      await fs.access(this.modelsDirectory);
    } catch {
      await fs.mkdir(this.modelsDirectory, { recursive: true });
    }
  }

  async createModel(name: string, config: ModelConfig): Promise<Model> {
    const insertModel: InsertModel = {
      name,
      config: config as any,
    };

    const model = await storage.createModel(insertModel);
    
    // Set model as ready immediately - Python initialization can happen later when needed
    const updatedModel = await storage.updateModel(model.id, { status: 'ready' });
    
    // Create model directory
    const modelDir = path.join(this.modelsDirectory, `model_${model.id}`);
    try {
      await fs.mkdir(modelDir, { recursive: true });
    } catch (error) {
      console.error(`Failed to create model directory: ${error}`);
    }

    return updatedModel;
  }

  async getModel(id: number): Promise<Model | undefined> {
    return await storage.getModel(id);
  }

  async getModels(): Promise<Model[]> {
    return await storage.getModels();
  }

  async updateModel(id: number, updates: Partial<Model>): Promise<Model> {
    return await storage.updateModel(id, updates);
  }

  async deleteModel(id: number): Promise<boolean> {
    const model = await storage.getModel(id);
    if (!model) return false;

    // Clean up model files
    const modelDir = path.join(this.modelsDirectory, `model_${id}`);
    try {
      await fs.rm(modelDir, { recursive: true, force: true });
    } catch (error) {
      console.error(`Failed to delete model directory: ${error}`);
    }

    return await storage.deleteModel(id);
  }

  async loadModel(id: number): Promise<void> {
    const model = await storage.getModel(id);
    if (!model) {
      throw new Error(`Model with id ${id} not found`);
    }

    if (model.status !== 'ready') {
      throw new Error(`Model ${id} is not ready for use`);
    }

    // Load model in Python
    await pythonBridge.initializeModel(model.config);
    this.activeModel = model;
    
    await storage.updateModel(id, { status: 'ready' });
  }

  getActiveModel(): Model | null {
    return this.activeModel;
  }

  async validateConfig(config: ModelConfig): Promise<boolean> {
    // Validate model configuration
    if (config.hidden_size % config.num_attention_heads !== 0) {
      throw new Error('hidden_size must be divisible by num_attention_heads');
    }

    if (config.head_dim && config.head_dim * config.num_attention_heads !== config.hidden_size) {
      throw new Error('head_dim * num_attention_heads must equal hidden_size');
    }

    if (config.use_moe && config.moe_config) {
      if (config.moe_config.num_experts_per_tok > config.moe_config.num_experts) {
        throw new Error('num_experts_per_tok cannot exceed num_experts');
      }
    }

    return true;
  }

  getModelPath(modelId: number): string {
    return path.join(this.modelsDirectory, `model_${modelId}`);
  }

  async getModelStats(id: number): Promise<any> {
    const model = await storage.getModel(id);
    if (!model) return null;

    const modelPath = this.getModelPath(id);
    let size = 0;
    
    try {
      const stats = await fs.stat(modelPath);
      size = stats.size;
    } catch {
      // Model directory might not exist yet
    }

    return {
      id: model.id,
      name: model.name,
      status: model.status,
      size,
      config: model.config,
      createdAt: model.createdAt,
      updatedAt: model.updatedAt,
    };
  }
}

export const modelManager = new ModelManager();
