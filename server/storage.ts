import { 
  users, 
  models, 
  trainingRuns, 
  checkpoints, 
  generations, 
  trainingMetrics,
  type User, 
  type InsertUser,
  type Model,
  type InsertModel,
  type TrainingRun,
  type InsertTrainingRun,
  type Checkpoint,
  type InsertCheckpoint,
  type Generation,
  type InsertGeneration,
  type TrainingMetric,
  type InsertTrainingMetric
} from "@shared/schema";

export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;

  // Model operations
  getModel(id: number): Promise<Model | undefined>;
  getModels(): Promise<Model[]>;
  createModel(model: InsertModel): Promise<Model>;
  updateModel(id: number, updates: Partial<Model>): Promise<Model>;
  deleteModel(id: number): Promise<boolean>;

  // Training run operations
  getTrainingRun(id: number): Promise<TrainingRun | undefined>;
  getTrainingRuns(modelId?: number): Promise<TrainingRun[]>;
  createTrainingRun(run: InsertTrainingRun): Promise<TrainingRun>;
  updateTrainingRun(id: number, updates: Partial<TrainingRun>): Promise<TrainingRun>;
  deleteTrainingRun(id: number): Promise<boolean>;

  // Checkpoint operations
  getCheckpoint(id: number): Promise<Checkpoint | undefined>;
  getCheckpoints(modelId?: number, trainingRunId?: number): Promise<Checkpoint[]>;
  createCheckpoint(checkpoint: InsertCheckpoint): Promise<Checkpoint>;
  updateCheckpoint(id: number, updates: Partial<Checkpoint>): Promise<Checkpoint>;
  deleteCheckpoint(id: number): Promise<boolean>;

  // Generation operations
  getGeneration(id: number): Promise<Generation | undefined>;
  getGenerations(modelId?: number, limit?: number): Promise<Generation[]>;
  createGeneration(generation: InsertGeneration): Promise<Generation>;
  deleteGeneration(id: number): Promise<boolean>;

  // Training metrics operations
  getTrainingMetrics(trainingRunId: number, limit?: number): Promise<TrainingMetric[]>;
  createTrainingMetric(metric: InsertTrainingMetric): Promise<TrainingMetric>;
  deleteTrainingMetrics(trainingRunId: number): Promise<boolean>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private models: Map<number, Model>;
  private trainingRuns: Map<number, TrainingRun>;
  private checkpoints: Map<number, Checkpoint>;
  private generations: Map<number, Generation>;
  private trainingMetrics: Map<number, TrainingMetric>;
  
  private currentUserId: number;
  private currentModelId: number;
  private currentTrainingRunId: number;
  private currentCheckpointId: number;
  private currentGenerationId: number;
  private currentMetricId: number;

  constructor() {
    this.users = new Map();
    this.models = new Map();
    this.trainingRuns = new Map();
    this.checkpoints = new Map();
    this.generations = new Map();
    this.trainingMetrics = new Map();
    
    this.currentUserId = 1;
    this.currentModelId = 1;
    this.currentTrainingRunId = 1;
    this.currentCheckpointId = 1;
    this.currentGenerationId = 1;
    this.currentMetricId = 1;
  }

  // User operations
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentUserId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  // Model operations
  async getModel(id: number): Promise<Model | undefined> {
    return this.models.get(id);
  }

  async getModels(): Promise<Model[]> {
    return Array.from(this.models.values());
  }

  async createModel(insertModel: InsertModel): Promise<Model> {
    const id = this.currentModelId++;
    const now = new Date();
    const model: Model = { 
      ...insertModel, 
      id, 
      status: "idle",
      createdAt: now,
      updatedAt: now
    };
    this.models.set(id, model);
    return model;
  }

  async updateModel(id: number, updates: Partial<Model>): Promise<Model> {
    const model = this.models.get(id);
    if (!model) {
      throw new Error(`Model with id ${id} not found`);
    }
    const updatedModel = { ...model, ...updates, updatedAt: new Date() };
    this.models.set(id, updatedModel);
    return updatedModel;
  }

  async deleteModel(id: number): Promise<boolean> {
    return this.models.delete(id);
  }

  // Training run operations
  async getTrainingRun(id: number): Promise<TrainingRun | undefined> {
    return this.trainingRuns.get(id);
  }

  async getTrainingRuns(modelId?: number): Promise<TrainingRun[]> {
    const runs = Array.from(this.trainingRuns.values());
    return modelId ? runs.filter(run => run.modelId === modelId) : runs;
  }

  async createTrainingRun(insertRun: InsertTrainingRun): Promise<TrainingRun> {
    const id = this.currentTrainingRunId++;
    const run: TrainingRun = { 
      ...insertRun, 
      id, 
      status: "pending",
      currentStep: 0,
      currentLoss: null,
      learningRate: null,
      startedAt: null,
      completedAt: null,
      createdAt: new Date(),
      modelId: insertRun.modelId ?? null
    };
    this.trainingRuns.set(id, run);
    return run;
  }

  async updateTrainingRun(id: number, updates: Partial<TrainingRun>): Promise<TrainingRun> {
    const run = this.trainingRuns.get(id);
    if (!run) {
      throw new Error(`Training run with id ${id} not found`);
    }
    const updatedRun = { ...run, ...updates };
    this.trainingRuns.set(id, updatedRun);
    return updatedRun;
  }

  async deleteTrainingRun(id: number): Promise<boolean> {
    return this.trainingRuns.delete(id);
  }

  // Checkpoint operations
  async getCheckpoint(id: number): Promise<Checkpoint | undefined> {
    return this.checkpoints.get(id);
  }

  async getCheckpoints(modelId?: number, trainingRunId?: number): Promise<Checkpoint[]> {
    let checkpoints = Array.from(this.checkpoints.values());
    if (modelId) {
      checkpoints = checkpoints.filter(cp => cp.modelId === modelId);
    }
    if (trainingRunId) {
      checkpoints = checkpoints.filter(cp => cp.trainingRunId === trainingRunId);
    }
    return checkpoints.sort((a, b) => (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0));
  }

  async createCheckpoint(insertCheckpoint: InsertCheckpoint): Promise<Checkpoint> {
    const id = this.currentCheckpointId++;
    const checkpoint: Checkpoint = { 
      ...insertCheckpoint, 
      id, 
      isBest: false,
      createdAt: new Date()
    };
    this.checkpoints.set(id, checkpoint);
    return checkpoint;
  }

  async updateCheckpoint(id: number, updates: Partial<Checkpoint>): Promise<Checkpoint> {
    const checkpoint = this.checkpoints.get(id);
    if (!checkpoint) {
      throw new Error(`Checkpoint with id ${id} not found`);
    }
    const updatedCheckpoint = { ...checkpoint, ...updates };
    this.checkpoints.set(id, updatedCheckpoint);
    return updatedCheckpoint;
  }

  async deleteCheckpoint(id: number): Promise<boolean> {
    return this.checkpoints.delete(id);
  }

  // Generation operations
  async getGeneration(id: number): Promise<Generation | undefined> {
    return this.generations.get(id);
  }

  async getGenerations(modelId?: number, limit?: number): Promise<Generation[]> {
    let generations = Array.from(this.generations.values());
    if (modelId) {
      generations = generations.filter(gen => gen.modelId === modelId);
    }
    generations.sort((a, b) => (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0));
    return limit ? generations.slice(0, limit) : generations;
  }

  async createGeneration(insertGeneration: InsertGeneration): Promise<Generation> {
    const id = this.currentGenerationId++;
    const generation: Generation = { 
      ...insertGeneration, 
      id, 
      createdAt: new Date()
    };
    this.generations.set(id, generation);
    return generation;
  }

  async deleteGeneration(id: number): Promise<boolean> {
    return this.generations.delete(id);
  }

  // Training metrics operations
  async getTrainingMetrics(trainingRunId: number, limit?: number): Promise<TrainingMetric[]> {
    let metrics = Array.from(this.trainingMetrics.values())
      .filter(metric => metric.trainingRunId === trainingRunId)
      .sort((a, b) => (a.step || 0) - (b.step || 0));
    return limit ? metrics.slice(-limit) : metrics;
  }

  async createTrainingMetric(insertMetric: InsertTrainingMetric): Promise<TrainingMetric> {
    const id = this.currentMetricId++;
    const metric: TrainingMetric = { 
      ...insertMetric, 
      id, 
      timestamp: new Date()
    };
    this.trainingMetrics.set(id, metric);
    return metric;
  }

  async deleteTrainingMetrics(trainingRunId: number): Promise<boolean> {
    let deleted = false;
    for (const [id, metric] of this.trainingMetrics.entries()) {
      if (metric.trainingRunId === trainingRunId) {
        this.trainingMetrics.delete(id);
        deleted = true;
      }
    }
    return deleted;
  }
}

export const storage = new MemStorage();
