import { 
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
import { promises as fs } from "fs";
import path from "path";

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

interface FileStorageData {
  users: User[];
  models: Model[];
  trainingRuns: TrainingRun[];
  checkpoints: Checkpoint[];
  generations: Generation[];
  trainingMetrics: TrainingMetric[];
  counters: {
    userId: number;
    modelId: number;
    trainingRunId: number;
    checkpointId: number;
    generationId: number;
    metricId: number;
  };
}

export class FileStorage implements IStorage {
  private dataDir: string;
  private dataFile: string;
  private data: FileStorageData;

  constructor() {
    this.dataDir = path.join(process.cwd(), 'data');
    this.dataFile = path.join(this.dataDir, 'storage.json');
    this.data = {
      users: [],
      models: [],
      trainingRuns: [],
      checkpoints: [],
      generations: [],
      trainingMetrics: [],
      counters: {
        userId: 1,
        modelId: 1,
        trainingRunId: 1,
        checkpointId: 1,
        generationId: 1,
        metricId: 1,
      }
    };
    this.loadData();
  }

  private async ensureDataDir(): Promise<void> {
    try {
      await fs.mkdir(this.dataDir, { recursive: true });
    } catch (error) {
      // Directory already exists or other error
    }
  }

  private async loadData(): Promise<void> {
    try {
      await this.ensureDataDir();
      const data = await fs.readFile(this.dataFile, 'utf8');
      this.data = JSON.parse(data);
      // Convert date strings back to Date objects
      this.data.users = this.data.users || [];
      this.data.models = (this.data.models || []).map(model => ({
        ...model,
        createdAt: model.createdAt ? new Date(model.createdAt) : null,
        updatedAt: model.updatedAt ? new Date(model.updatedAt) : null,
      }));
      this.data.trainingRuns = (this.data.trainingRuns || []).map(run => ({
        ...run,
        createdAt: run.createdAt ? new Date(run.createdAt) : null,
        startedAt: run.startedAt ? new Date(run.startedAt) : null,
        completedAt: run.completedAt ? new Date(run.completedAt) : null,
      }));
      this.data.checkpoints = (this.data.checkpoints || []).map(cp => ({
        ...cp,
        createdAt: cp.createdAt ? new Date(cp.createdAt) : null,
      }));
      this.data.generations = (this.data.generations || []).map(gen => ({
        ...gen,
        createdAt: gen.createdAt ? new Date(gen.createdAt) : null,
      }));
      this.data.trainingMetrics = (this.data.trainingMetrics || []).map(metric => ({
        ...metric,
        timestamp: metric.timestamp ? new Date(metric.timestamp) : null,
      }));
      this.data.counters = this.data.counters || {
        userId: 1,
        modelId: 1,
        trainingRunId: 1,
        checkpointId: 1,
        generationId: 1,
        metricId: 1,
      };
    } catch (error) {
      // File doesn't exist or invalid JSON, use default data
      console.log('Loading default data structure');
    }
  }

  private async saveData(): Promise<void> {
    try {
      await this.ensureDataDir();
      await fs.writeFile(this.dataFile, JSON.stringify(this.data, null, 2));
    } catch (error) {
      console.error('Failed to save data:', error);
    }
  }

  // User operations
  async getUser(id: number): Promise<User | undefined> {
    return this.data.users.find(user => user.id === id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return this.data.users.find(user => user.username === username);
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const user: User = {
      ...insertUser,
      id: this.data.counters.userId++,
    };
    this.data.users.push(user);
    await this.saveData();
    return user;
  }

  // Model operations
  async getModel(id: number): Promise<Model | undefined> {
    return this.data.models.find(model => model.id === id);
  }

  async getModels(): Promise<Model[]> {
    return [...this.data.models].sort((a, b) => 
      (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0)
    );
  }

  async createModel(insertModel: InsertModel): Promise<Model> {
    const now = new Date();
    const model: Model = {
      ...insertModel,
      id: this.data.counters.modelId++,
      status: "idle",
      createdAt: now,
      updatedAt: now,
    };
    this.data.models.push(model);
    await this.saveData();
    return model;
  }

  async updateModel(id: number, updates: Partial<Model>): Promise<Model> {
    const modelIndex = this.data.models.findIndex(model => model.id === id);
    if (modelIndex === -1) {
      throw new Error(`Model with id ${id} not found`);
    }
    const updatedModel = {
      ...this.data.models[modelIndex],
      ...updates,
      updatedAt: new Date(),
    };
    this.data.models[modelIndex] = updatedModel;
    await this.saveData();
    return updatedModel;
  }

  async deleteModel(id: number): Promise<boolean> {
    const initialLength = this.data.models.length;
    this.data.models = this.data.models.filter(model => model.id !== id);
    if (this.data.models.length < initialLength) {
      await this.saveData();
      return true;
    }
    return false;
  }

  // Training run operations
  async getTrainingRun(id: number): Promise<TrainingRun | undefined> {
    return this.data.trainingRuns.find(run => run.id === id);
  }

  async getTrainingRuns(modelId?: number): Promise<TrainingRun[]> {
    let runs = [...this.data.trainingRuns];
    if (modelId) {
      runs = runs.filter(run => run.modelId === modelId);
    }
    return runs.sort((a, b) => 
      (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0)
    );
  }

  async createTrainingRun(insertRun: InsertTrainingRun): Promise<TrainingRun> {
    const run: TrainingRun = {
      ...insertRun,
      id: this.data.counters.trainingRunId++,
      status: "pending",
      currentStep: 0,
      currentLoss: null,
      learningRate: null,
      startedAt: null,
      completedAt: null,
      createdAt: new Date(),
    };
    this.data.trainingRuns.push(run);
    await this.saveData();
    return run;
  }

  async updateTrainingRun(id: number, updates: Partial<TrainingRun>): Promise<TrainingRun> {
    const runIndex = this.data.trainingRuns.findIndex(run => run.id === id);
    if (runIndex === -1) {
      throw new Error(`Training run with id ${id} not found`);
    }
    const updatedRun = { ...this.data.trainingRuns[runIndex], ...updates };
    this.data.trainingRuns[runIndex] = updatedRun;
    await this.saveData();
    return updatedRun;
  }

  async deleteTrainingRun(id: number): Promise<boolean> {
    const initialLength = this.data.trainingRuns.length;
    this.data.trainingRuns = this.data.trainingRuns.filter(run => run.id !== id);
    if (this.data.trainingRuns.length < initialLength) {
      await this.saveData();
      return true;
    }
    return false;
  }

  // Checkpoint operations
  async getCheckpoint(id: number): Promise<Checkpoint | undefined> {
    return this.data.checkpoints.find(checkpoint => checkpoint.id === id);
  }

  async getCheckpoints(modelId?: number, trainingRunId?: number): Promise<Checkpoint[]> {
    let checkpoints = [...this.data.checkpoints];
    if (modelId) {
      checkpoints = checkpoints.filter(cp => cp.modelId === modelId);
    }
    if (trainingRunId) {
      checkpoints = checkpoints.filter(cp => cp.trainingRunId === trainingRunId);
    }
    return checkpoints.sort((a, b) => (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0));
  }

  async createCheckpoint(insertCheckpoint: InsertCheckpoint): Promise<Checkpoint> {
    const checkpoint: Checkpoint = {
      ...insertCheckpoint,
      id: this.data.counters.checkpointId++,
      isBest: false,
      createdAt: new Date()
    };
    this.data.checkpoints.push(checkpoint);
    await this.saveData();
    return checkpoint;
  }

  async updateCheckpoint(id: number, updates: Partial<Checkpoint>): Promise<Checkpoint> {
    const checkpointIndex = this.data.checkpoints.findIndex(cp => cp.id === id);
    if (checkpointIndex === -1) {
      throw new Error(`Checkpoint with id ${id} not found`);
    }
    const updatedCheckpoint = { ...this.data.checkpoints[checkpointIndex], ...updates };
    this.data.checkpoints[checkpointIndex] = updatedCheckpoint;
    await this.saveData();
    return updatedCheckpoint;
  }

  async deleteCheckpoint(id: number): Promise<boolean> {
    const initialLength = this.data.checkpoints.length;
    this.data.checkpoints = this.data.checkpoints.filter(cp => cp.id !== id);
    if (this.data.checkpoints.length < initialLength) {
      await this.saveData();
      return true;
    }
    return false;
  }

  // Generation operations
  async getGeneration(id: number): Promise<Generation | undefined> {
    return this.data.generations.find(generation => generation.id === id);
  }

  async getGenerations(modelId?: number, limit?: number): Promise<Generation[]> {
    let generations = [...this.data.generations];
    if (modelId) {
      generations = generations.filter(gen => gen.modelId === modelId);
    }
    generations.sort((a, b) => (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0));
    return limit ? generations.slice(0, limit) : generations;
  }

  async createGeneration(insertGeneration: InsertGeneration): Promise<Generation> {
    const generation: Generation = {
      ...insertGeneration,
      id: this.data.counters.generationId++,
      createdAt: new Date()
    };
    this.data.generations.push(generation);
    await this.saveData();
    return generation;
  }

  async deleteGeneration(id: number): Promise<boolean> {
    const initialLength = this.data.generations.length;
    this.data.generations = this.data.generations.filter(gen => gen.id !== id);
    if (this.data.generations.length < initialLength) {
      await this.saveData();
      return true;
    }
    return false;
  }

  // Training metrics operations
  async getTrainingMetrics(trainingRunId: number, limit?: number): Promise<TrainingMetric[]> {
    let metrics = this.data.trainingMetrics
      .filter(metric => metric.trainingRunId === trainingRunId)
      .sort((a, b) => (a.step || 0) - (b.step || 0));
    return limit ? metrics.slice(-limit) : metrics;
  }

  async createTrainingMetric(insertMetric: InsertTrainingMetric): Promise<TrainingMetric> {
    const metric: TrainingMetric = {
      ...insertMetric,
      id: this.data.counters.metricId++,
      timestamp: new Date()
    };
    this.data.trainingMetrics.push(metric);
    await this.saveData();
    return metric;
  }

  async deleteTrainingMetrics(trainingRunId: number): Promise<boolean> {
    const initialLength = this.data.trainingMetrics.length;
    this.data.trainingMetrics = this.data.trainingMetrics.filter(
      metric => metric.trainingRunId !== trainingRunId
    );
    if (this.data.trainingMetrics.length < initialLength) {
      await this.saveData();
      return true;
    }
    return false;
  }
}

export const storage = new FileStorage();
