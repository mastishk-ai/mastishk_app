import { pgTable, text, serial, integer, boolean, jsonb, timestamp, real } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const models = pgTable("models", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  config: jsonb("config").notNull(),
  status: text("status").notNull().default("idle"), // idle, training, ready, error
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const trainingRuns = pgTable("training_runs", {
  id: serial("id").primaryKey(),
  modelId: integer("model_id").references(() => models.id),
  name: text("name").notNull(),
  config: jsonb("config").notNull(),
  status: text("status").notNull().default("pending"), // pending, running, completed, failed, stopped
  currentStep: integer("current_step").default(0),
  totalSteps: integer("total_steps").notNull(),
  currentLoss: real("current_loss"),
  learningRate: real("learning_rate"),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const checkpoints = pgTable("checkpoints", {
  id: serial("id").primaryKey(),
  modelId: integer("model_id").references(() => models.id),
  trainingRunId: integer("training_run_id").references(() => trainingRuns.id),
  name: text("name").notNull(),
  step: integer("step").notNull(),
  loss: real("loss"),
  filePath: text("file_path").notNull(),
  fileSize: integer("file_size"),
  metadata: jsonb("metadata"),
  isBest: boolean("is_best").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

export const generations = pgTable("generations", {
  id: serial("id").primaryKey(),
  modelId: integer("model_id").references(() => models.id),
  prompt: text("prompt").notNull(),
  output: text("output").notNull(),
  config: jsonb("config").notNull(),
  tokensGenerated: integer("tokens_generated"),
  generationTime: real("generation_time"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const trainingMetrics = pgTable("training_metrics", {
  id: serial("id").primaryKey(),
  trainingRunId: integer("training_run_id").references(() => trainingRuns.id),
  step: integer("step").notNull(),
  loss: real("loss").notNull(),
  learningRate: real("learning_rate").notNull(),
  gpuUtilization: real("gpu_utilization"),
  memoryUsage: real("memory_usage"),
  expertUtilization: jsonb("expert_utilization"),
  layerSkipRate: real("layer_skip_rate"),
  timestamp: timestamp("timestamp").defaultNow(),
});

// Insert schemas
export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertModelSchema = createInsertSchema(models).pick({
  name: true,
  config: true,
});

export const insertTrainingRunSchema = createInsertSchema(trainingRuns).pick({
  modelId: true,
  name: true,
  config: true,
  totalSteps: true,
});

export const insertCheckpointSchema = createInsertSchema(checkpoints).pick({
  modelId: true,
  trainingRunId: true,
  name: true,
  step: true,
  loss: true,
  filePath: true,
  fileSize: true,
  metadata: true,
});

export const insertGenerationSchema = createInsertSchema(generations).pick({
  modelId: true,
  prompt: true,
  output: true,
  config: true,
  tokensGenerated: true,
  generationTime: true,
});

export const insertTrainingMetricSchema = createInsertSchema(trainingMetrics).pick({
  trainingRunId: true,
  step: true,
  loss: true,
  learningRate: true,
  gpuUtilization: true,
  memoryUsage: true,
  expertUtilization: true,
  layerSkipRate: true,
});

// Types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export type InsertModel = z.infer<typeof insertModelSchema>;
export type Model = typeof models.$inferSelect;

export type InsertTrainingRun = z.infer<typeof insertTrainingRunSchema>;
export type TrainingRun = typeof trainingRuns.$inferSelect;

export type InsertCheckpoint = z.infer<typeof insertCheckpointSchema>;
export type Checkpoint = typeof checkpoints.$inferSelect;

export type InsertGeneration = z.infer<typeof insertGenerationSchema>;
export type Generation = typeof generations.$inferSelect;

export type InsertTrainingMetric = z.infer<typeof insertTrainingMetricSchema>;
export type TrainingMetric = typeof trainingMetrics.$inferSelect;

// Configuration schemas
export const ModelConfigSchema = z.object({
  // Core architecture
  vocab_size: z.number().min(1000).max(1000000).default(32000),
  hidden_size: z.number().min(512).max(16384).default(4096),
  intermediate_size: z.number().min(1024).max(65536).default(11008),
  num_hidden_layers: z.number().min(6).max(96).default(32),
  num_attention_heads: z.number().min(8).max(128).default(32),
  num_key_value_heads: z.number().min(1).max(128).default(8),
  head_dim: z.number().optional(),
  hidden_act: z.string().default("silu"),
  max_position_embeddings: z.number().min(512).max(262144).default(4096),
  initializer_range: z.number().default(0.02),
  rms_norm_eps: z.number().default(1e-5),
  learning_rate: z.number().min(1e-6).max(1e-1).default(5e-4),
  
  // Advanced features
  use_flash_attention: z.boolean().default(true),
  use_differential_attention: z.boolean().default(false),
  differential_lambda_init: z.number().default(0.5),
  use_minimax: z.boolean().default(false),
  minimax_layer_frequency: z.number().default(4),
  minimax_adversarial_epsilon: z.number().default(0.1),
  minimax_iterations: z.number().default(3),
  lolcats_enabled: z.boolean().default(false),
  lolcats_compression_dim: z.number().default(512),
  use_multi_token_prediction: z.boolean().default(false),
  n_predict_tokens: z.number().default(4),
  
  // MoE configuration
  use_moe: z.boolean().default(false),
  moe_config: z.object({
    num_experts: z.number().min(2).max(32).default(8),
    num_experts_per_tok: z.number().min(1).max(8).default(2),
    expert_capacity_factor: z.number().default(1.25),
    aux_loss_weight: z.number().default(0.01),
    router_z_loss_weight: z.number().default(0.001),
    router_dropout: z.number().default(0.1),
    expert_dropout: z.number().default(0.1),
    moe_layer_frequency: z.number().default(2),
    load_balancing_type: z.string().default("aux_loss"),
    router_type: z.string().default("top_k"),
  }).optional(),
  
  // MoD configuration
  use_mod: z.boolean().default(false),
  mod_config: z.object({
    enabled: z.boolean().default(true),
    router_type: z.string().default("learned"),
    skip_probability: z.number().min(0).max(0.5).default(0.2),
    min_layers_per_token: z.number().default(12),
    capacity_factor: z.number().min(0.1).max(1.0).default(0.8),
    router_aux_loss_weight: z.number().default(0.01),
    router_z_loss_weight: z.number().default(0.001),
    load_balancing_type: z.string().default("auxiliary"),
    router_hidden_dim: z.number().default(256),
    router_dropout: z.number().default(0.1),
    temperature: z.number().default(1.0),
    use_gumbel_softmax: z.boolean().default(true),
    straight_through: z.boolean().default(true),
    block_size: z.number().default(1),
  }).optional(),
});

export const TrainingConfigSchema = z.object({
  learning_rate: z.number().min(1e-6).max(1e-2).default(5e-4),
  batch_size: z.number().min(1).max(128).default(2),
  gradient_accumulation_steps: z.number().min(1).max(64).default(4),
  max_steps: z.number().min(1).max(1000000).default(10000),
  eval_steps: z.number().min(1).max(10000).default(500),
  save_steps: z.number().min(1).max(10000).default(1000),
  warmup_steps: z.number().min(0).max(10000).default(100),
  max_grad_norm: z.number().min(0).max(10).default(1.0),
  weight_decay: z.number().min(0).max(1).default(0.01),
  mixed_precision: z.boolean().default(true),
  gradient_checkpointing: z.boolean().default(true),
  early_stopping: z.boolean().default(false),
  early_stopping_patience: z.number().default(5),
  early_stopping_threshold: z.number().default(1e-4),
  use_wandb: z.boolean().default(false),
  use_tensorboard: z.boolean().default(false),
  seed: z.number().default(42),
  
  // Enhanced checkpoint settings
  save_optimizer_state: z.boolean().default(true),
  save_scheduler_state: z.boolean().default(true),
  save_random_states: z.boolean().default(true),
  verify_integrity: z.boolean().default(true),
  max_checkpoints: z.number().default(100),
  auto_save_interval: z.number().default(1000),
  
  // Weight logging and verification
  enable_weight_logging: z.boolean().default(false),
  weight_verification: z.boolean().default(false),
});

export const GenerationConfigSchema = z.object({
  temperature: z.number().min(0.1).max(2.0).default(0.7),
  top_p: z.number().min(0.1).max(1.0).default(0.9),
  top_k: z.number().min(1).max(100).default(50),
  max_length: z.number().min(10).max(2048).default(500),
  repetition_penalty: z.number().min(1.0).max(2.0).default(1.1),
  length_penalty: z.number().min(0.5).max(2.0).default(1.0),
  no_repeat_ngram_size: z.number().min(0).max(10).default(3),
  do_sample: z.boolean().default(true),
  early_stopping: z.boolean().default(false),
  num_beams: z.number().min(1).max(10).default(1),
  use_multi_token_prediction: z.boolean().default(false),
});

export type ModelConfig = z.infer<typeof ModelConfigSchema>;
export type TrainingConfig = z.infer<typeof TrainingConfigSchema>;
export type GenerationConfig = z.infer<typeof GenerationConfigSchema>;
