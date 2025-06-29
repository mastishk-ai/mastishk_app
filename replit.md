# Mastishk Transformer Studio

## Overview

Mastishk Transformer Studio is an advanced transformer experimentation platform that provides a comprehensive environment for training, configuring, and experimenting with sophisticated transformer models. The application features a modern web interface built with React and shadcn/ui components, paired with a Node.js/Express backend that communicates with Python-based machine learning services for model operations.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript
- **UI Library**: shadcn/ui components with Radix UI primitives
- **Styling**: Tailwind CSS with custom design tokens
- **State Management**: TanStack Query for server state, React hooks for local state
- **Routing**: Wouter for client-side routing
- **Real-time Communication**: WebSocket integration for live training updates

### Backend Architecture
- **Server**: Node.js with Express.js
- **Language**: TypeScript with ES modules
- **API Design**: RESTful endpoints with WebSocket support
- **File Handling**: Multer for training data uploads
- **Python Integration**: Child process communication with Python ML services

### Data Storage Solutions
- **Database**: PostgreSQL with Drizzle ORM
- **Schema Management**: Drizzle Kit for migrations
- **Connection**: Neon Database serverless driver
- **File Storage**: Local filesystem for model checkpoints and training data

## Key Components

### Model Configuration System
- Advanced transformer architecture configuration including MoE (Mixture of Experts) and MoD (Mixture of Depths)
- Support for differential attention, MiniMax optimization, and LoLCATs compression
- Multi-token prediction capabilities
- Flash attention integration for performance optimization

### Training Pipeline
- Comprehensive training configuration with optimizer and scheduler state management
- Data upload system supporting multiple formats (TXT, JSON, JSONL, CSV)
- Real-time training monitoring with loss curves and system metrics
- Checkpoint management with integrity verification

### Text Generation Engine
- Advanced generation strategies including beam search and nucleus sampling
- Configurable parameters for temperature, top-p, top-k sampling
- Multi-token prediction support
- Generation history tracking and export capabilities

### Monitoring and Analytics
- Real-time training metrics visualization
- GPU utilization and memory monitoring
- Expert utilization tracking for MoE models
- Comprehensive performance analytics dashboard

## Data Flow

1. **Model Creation**: Users configure transformer parameters through the web interface
2. **Training Data**: Files are uploaded via the web UI and processed by the backend
3. **Training Execution**: Backend spawns Python processes for model training
4. **Real-time Updates**: WebSocket connections provide live training progress
5. **Checkpoint Management**: Model states are automatically saved and managed
6. **Text Generation**: Trained models are used for inference through the web interface

## External Dependencies

### Core Technologies
- **Database**: PostgreSQL (configured for Neon serverless)
- **Python ML Stack**: PyTorch, Transformers, Flash Attention
- **UI Components**: Radix UI primitives, Recharts for visualizations
- **Development Tools**: Vite for build tooling, ESBuild for production

### Python Dependencies
- PyTorch ecosystem for deep learning
- Transformers library for model implementations
- Flash Attention for optimized attention mechanisms
- Streamlit integration for advanced experimentation

## Deployment Strategy

### Development Environment
- Vite development server with HMR (Hot Module Replacement)
- TypeScript compilation with strict mode enabled
- Real-time error overlay integration
- WebSocket proxy for development

### Production Build
- Vite build process for optimized frontend assets
- ESBuild bundling for Node.js backend
- Static asset serving from dist/public directory
- Environment-based configuration management

### Database Management
- Drizzle migrations for schema versioning
- Connection pooling for scalability
- Automatic schema synchronization

## Changelog

- June 29, 2025: Fixed Start Training button functionality and added learning rate parameter
  - Resolved "Model not ready for training" errors by changing default model status from "idle" to "ready"
  - Added learning rate parameter with default value 5.0e-4 and range control (1e-6 to 1e-1)
  - Implemented mock training simulation for cases where Python bridge is unavailable
  - Updated model configuration components to include learning rate in architecture settings
  - Fixed model status validation in training manager to properly check for ready models
- June 29, 2025: Implemented file-based persistent storage system to fix data loss issues
  - Replaced volatile in-memory storage with JSON file-based storage in 'data' folder
  - Models now persist across server restarts and are saved to storage.json
  - Enhanced cache invalidation with detailed logging for debugging dropdown issues
  - Added comprehensive mutation debugging to track model creation and cache updates
- June 29, 2025: Added comprehensive documentation system
  - Created complete Documentation page with all app features and usage guides
  - Added API reference, troubleshooting guide, and system architecture details
  - Integrated documentation into main navigation for easy access
  - Included getting started guide, best practices, and testing information
- June 29, 2025: Updated branding and copyright information
  - Changed application name from "AI Transformer Platform" to "Mastishk" with bold styling
  - Added "Transformer Studio" subtitle for clarity
  - Integrated copyright notice "© 2025 Aman Sharma" in sidebar and header
  - Enhanced branding consistency across all components
- June 29, 2025: Fixed text alignment and overflow issues throughout application
  - Resolved Quick Presets section text overflow problems
  - Added comprehensive CSS utilities for text truncation and line clamping
  - Implemented responsive design improvements for better mobile/tablet display
  - Enhanced navigation items with proper text containment
- June 29, 2025: Implemented premium UI design system
  - Created modern glass-effect design with premium styling throughout application
  - Enhanced color scheme with sophisticated purple-blue gradient primary colors
  - Added smooth animations, hover effects, and premium card components
  - Improved typography with gradient text effects and better spacing
  - Redesigned sidebar with professional branding and enhanced navigation
  - Updated theme toggle with premium styling and smooth transitions
  - Applied consistent premium design language across all components
- June 29, 2025: Implemented app-wide theme switching system
  - Added dark/light mode theme toggle to all pages (Dashboard, Visualization)
  - Enhanced 3D visualizations with theme-adaptive colors for better visibility
  - Fixed Plotly chart background, axis, and text colors to respond to theme changes
  - Implemented proper theme initialization and system preference detection
  - Resolved blank generative transformer architecture visualization display
- June 29, 2025: Enhanced 3D generative transformer architecture visualization
  - Created specialized GenerativeArchitecture3D component for transformer visualization
  - Added three interactive view modes: Overview, Layer-by-layer, and Token Flow
  - Implemented animation controls for layer progression exploration
  - Enhanced visual representation with token flow, attention rings, and MLP chains
  - Added architecture details panel with real-time parameter display
  - Integrated as third tab in visualization page for comprehensive 3D architecture viewing
- June 29, 2025: Enhanced checkpoint management system with complete state preservation
  - Optimizer momentum and state tracking for training continuity
  - Learning rate scheduler state preservation
  - Random states preservation for reproducibility
  - Weight update verification system
  - Integrity checking for checkpoint files
  - Auto-save interval configuration
  - Comprehensive 3D visualization system integrated from Python script
- June 28, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.