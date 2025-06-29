import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import multer, { FileFilterCallback } from "multer";
import path from "path";
import fs from "fs";
import type { Request } from "express";

import { storage } from "./storage";
import { modelManager } from "./services/model-manager";
import { trainingManager } from "./services/training-manager";
import { checkpointManager } from "./services/checkpoint-manager";
import { pythonBridge } from "./services/python-bridge";
import crawlRoutes from './crawl.js';

import { 
  insertModelSchema, 
  insertTrainingRunSchema, 
  insertGenerationSchema,
  ModelConfigSchema,
  TrainingConfigSchema,
  GenerationConfigSchema
} from "@shared/schema";

// Configure multer for file uploads
const upload = multer({
  storage: multer.diskStorage({
    destination: (req: Request, file: Express.Multer.File, cb: (error: Error | null, destination: string) => void) => {
      const uploadDir = path.join(process.cwd(), 'uploads');
      if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir, { recursive: true });
      }
      cb(null, uploadDir);
    },
    filename: (req: Request, file: Express.Multer.File, cb: (error: Error | null, filename: string) => void) => {
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
      cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
  }),
  fileFilter: (req: Request, file: Express.Multer.File, cb: FileFilterCallback) => {
    const allowedTypes = ['.txt', '.json', '.jsonl', '.csv'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`File type ${ext} not allowed`));
    }
  },
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit
  }
});

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  
  // Priority API middleware - handle before any other middleware
  app.use((req, res, next) => {
    if (req.path.startsWith('/api/')) {
      console.log(`API Request: ${req.method} ${req.path}`);
      // Ensure JSON responses for API routes
      res.setHeader('Content-Type', 'application/json');
    }
    next();
  });

  // WebSocket server for real-time updates
  const wss = new WebSocketServer({ server: httpServer, path: '/ws' });
  
  const connectedClients = new Set<WebSocket>();

  wss.on('connection', (ws) => {
    connectedClients.add(ws);
    console.log('Client connected to WebSocket');

    ws.on('close', () => {
      connectedClients.delete(ws);
      console.log('Client disconnected from WebSocket');
    });

    ws.on('error', (error) => {
      console.error('WebSocket error:', error);
      connectedClients.delete(ws);
    });
  });

  // Broadcast function for real-time updates
  const broadcast = (message: any) => {
    const messageStr = JSON.stringify(message);
    connectedClients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(messageStr);
      }
    });
  };

  // Set up training manager event listeners for real-time updates
  trainingManager.on('training_progress', (data) => {
    broadcast({ type: 'training_progress', data });
  });

  trainingManager.on('training_complete', (data) => {
    broadcast({ type: 'training_complete', data });
  });

  trainingManager.on('training_error', (data) => {
    broadcast({ type: 'training_error', data });
  });

  // Test endpoint
  app.post('/api/test', (req, res) => {
    console.log('Test endpoint hit with body:', req.body);
    res.json({ success: true, body: req.body });
  });

  // Model management routes
  app.get('/api/models', async (req, res) => {
    try {
      const models = await modelManager.getModels();
      res.json(models);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch models' });
    }
  });

  app.get('/api/models/:id', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const model = await modelManager.getModel(id);
      if (!model) {
        return res.status(404).json({ error: 'Model not found' });
      }
      res.json(model);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch model' });
    }
  });

  app.post('/api/models', async (req, res) => {
    try {
      console.log('Model creation endpoint hit');
      
      // Ensure this is a proper API request
      if (!req.body || typeof req.body !== 'object') {
        console.log('Invalid request body');
        return res.status(400).json({ error: 'Invalid request body' });
      }
      
      const { name, config } = req.body;
      
      if (!name || !config) {
        console.log('Missing required fields');
        return res.status(400).json({ error: 'Missing required fields: name and config' });
      }
      
      console.log(`Creating model: ${name}`);
      
      // Create model directly
      const insertModel = { name, config };
      const model = await storage.createModel(insertModel);
      
      console.log(`Model created successfully: ID ${model.id}, Name: ${model.name}`);
      
      // Return JSON response
      return res.status(201).json(model);
    } catch (error) {
      console.error('Model creation error:', error);
      return res.status(500).json({ 
        error: error instanceof Error ? error.message : 'Failed to create model' 
      });
    }
  });

  app.put('/api/models/:id', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const updates = req.body;
      const model = await modelManager.updateModel(id, updates);
      res.json(model);
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : 'Failed to update model' });
    }
  });

  app.delete('/api/models/:id', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const success = await modelManager.deleteModel(id);
      if (!success) {
        return res.status(404).json({ error: 'Model not found' });
      }
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: 'Failed to delete model' });
    }
  });

  app.post('/api/models/:id/load', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      await modelManager.loadModel(id);
      res.json({ success: true });
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : 'Failed to load model' });
    }
  });

  app.get('/api/models/:id/stats', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const stats = await modelManager.getModelStats(id);
      if (!stats) {
        return res.status(404).json({ error: 'Model not found' });
      }
      res.json(stats);
    } catch (error) {
      res.status(500).json({ error: 'Failed to get model stats' });
    }
  });

  // Training routes
  app.get('/api/training-runs', async (req, res) => {
    try {
      const modelId = req.query.modelId ? parseInt(req.query.modelId as string) : undefined;
      const runs = await trainingManager.getTrainingRuns(modelId);
      res.json(runs);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch training runs' });
    }
  });

  app.get('/api/training-runs/:id', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const run = await trainingManager.getTrainingRun(id);
      if (!run) {
        return res.status(404).json({ error: 'Training run not found' });
      }
      res.json(run);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch training run' });
    }
  });

  app.post('/api/training-runs', async (req, res) => {
    try {
      const { modelId, name, config, dataFiles } = req.body;
      const trainingConfig = TrainingConfigSchema.parse(config);
      
      const run = await trainingManager.startTraining(modelId, name, trainingConfig, dataFiles || []);
      res.status(201).json(run);
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : 'Failed to start training' });
    }
  });

  app.post('/api/training-runs/:id/stop', async (req, res) => {
    try {
      await trainingManager.stopTraining();
      res.json({ success: true });
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : 'Failed to stop training' });
    }
  });

  app.post('/api/training-runs/:id/pause', async (req, res) => {
    try {
      await trainingManager.pauseTraining();
      res.json({ success: true });
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : 'Failed to pause training' });
    }
  });

  app.post('/api/training-runs/:id/resume', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      await trainingManager.resumeTraining(id);
      res.json({ success: true });
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : 'Failed to resume training' });
    }
  });

  app.get('/api/training-runs/:id/metrics', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const limit = req.query.limit ? parseInt(req.query.limit as string) : undefined;
      const metrics = await trainingManager.getTrainingMetrics(id, limit);
      res.json(metrics);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch training metrics' });
    }
  });

  app.get('/api/training/status', async (req, res) => {
    try {
      const activeRun = trainingManager.getActiveTrainingRun();
      res.json({
        isTraining: !!activeRun,
        activeRun,
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to get training status' });
    }
  });

  // File upload routes
  app.post('/api/upload/training-data', upload.array('files'), async (req, res) => {
    try {
      const files = req.files as Express.Multer.File[];
      if (!files || files.length === 0) {
        return res.status(400).json({ error: 'No files uploaded' });
      }

      const uploadedFiles = files.map(file => ({
        filename: file.filename,
        originalName: file.originalname,
        path: file.path,
        size: file.size,
      }));

      res.json({ files: uploadedFiles });
    } catch (error) {
      res.status(500).json({ error: 'Failed to upload files' });
    }
  });

  // Text generation routes
  app.post('/api/generate', async (req, res) => {
    try {
      const { modelId, prompt, config } = req.body;
      const generationConfig = GenerationConfigSchema.parse(config);
      
      // Ensure model is loaded
      const activeModel = modelManager.getActiveModel();
      if (!activeModel || activeModel.id !== modelId) {
        await modelManager.loadModel(modelId);
      }

      // Generate text
      const result = await pythonBridge.generateText(prompt, generationConfig);
      
      // Save generation
      const generation = await storage.createGeneration({
        modelId,
        prompt,
        output: result.output,
        config: generationConfig as any,
        tokensGenerated: result.tokensGenerated,
        generationTime: result.generationTime,
      });

      res.json({
        ...result,
        id: generation.id,
      });
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : 'Failed to generate text' });
    }
  });

  app.get('/api/generations', async (req, res) => {
    try {
      const modelId = req.query.modelId ? parseInt(req.query.modelId as string) : undefined;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 20;
      const generations = await storage.getGenerations(modelId, limit);
      res.json(generations);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch generations' });
    }
  });

  // Checkpoint routes
  app.get('/api/checkpoints', async (req, res) => {
    try {
      const modelId = req.query.modelId ? parseInt(req.query.modelId as string) : undefined;
      const trainingRunId = req.query.trainingRunId ? parseInt(req.query.trainingRunId as string) : undefined;
      const checkpoints = await checkpointManager.getCheckpoints(modelId, trainingRunId);
      res.json(checkpoints);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch checkpoints' });
    }
  });

  app.get('/api/checkpoints/:id', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const checkpoint = await checkpointManager.getCheckpoint(id);
      if (!checkpoint) {
        return res.status(404).json({ error: 'Checkpoint not found' });
      }
      res.json(checkpoint);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch checkpoint' });
    }
  });

  app.post('/api/checkpoints/:id/load', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      await checkpointManager.loadCheckpoint(id);
      res.json({ success: true });
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : 'Failed to load checkpoint' });
    }
  });

  app.delete('/api/checkpoints/:id', async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const success = await checkpointManager.deleteCheckpoint(id);
      if (!success) {
        return res.status(404).json({ error: 'Checkpoint not found' });
      }
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: 'Failed to delete checkpoint' });
    }
  });

  app.post('/api/checkpoints/cleanup', async (req, res) => {
    try {
      const { modelId, maxCheckpoints } = req.body;
      const deletedCount = await checkpointManager.cleanupOldCheckpoints(modelId, maxCheckpoints);
      res.json({ deletedCount });
    } catch (error) {
      res.status(500).json({ error: 'Failed to cleanup checkpoints' });
    }
  });

  app.get('/api/checkpoints/stats', async (req, res) => {
    try {
      const modelId = req.query.modelId ? parseInt(req.query.modelId as string) : undefined;
      const stats = await checkpointManager.getCheckpointStats(modelId);
      res.json(stats);
    } catch (error) {
      res.status(500).json({ error: 'Failed to get checkpoint stats' });
    }
  });

  // Health check
  app.get('/api/health', (req, res) => {
    res.json({ 
      status: 'ok', 
      timestamp: new Date().toISOString(),
      services: {
        pythonBridge: pythonBridge ? 'connected' : 'disconnected',
        training: trainingManager.getActiveTrainingRun() ? 'active' : 'idle',
      }
    });
  });

  return httpServer;
}
