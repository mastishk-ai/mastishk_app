import { useState } from "react";
import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { 
  Settings, 
  Play, 
  Edit3, 
  Activity, 
  Save, 
  BarChart2, 
  Cpu,
  Zap,
  Box
} from "lucide-react";
import logoImage from "@assets/Copilot_20250629_034156_1751193748480.png";

interface SidebarProps {
  modelStatus?: {
    status: string;
    lastTrained?: string;
  };
}

export function Sidebar({ modelStatus }: SidebarProps) {
  const [location] = useLocation();

  const navigationItems = [
    {
      path: "/",
      label: "Model Configuration",
      icon: Settings,
      description: "Configure transformer parameters"
    },
    {
      path: "/training",
      label: "Training Pipeline", 
      icon: Play,
      description: "Upload data and manage training"
    },
    {
      path: "/generation",
      label: "Chat",
      icon: Edit3,
      description: "Chat with your AI model"
    },
    {
      path: "/monitoring", 
      label: "Training Monitor",
      icon: Activity,
      description: "Real-time training metrics"
    },
    {
      path: "/checkpoints",
      label: "Checkpoints",
      icon: Save,
      description: "Manage model checkpoints"
    },
    {
      path: "/analytics",
      label: "Analytics", 
      icon: BarChart2,
      description: "Performance analysis"
    },
    {
      path: "/testing",
      label: "Testing Suite",
      icon: Zap,
      description: "Component functionality testing"
    },
    {
      path: "/docs",
      label: "Documentation",
      icon: Box,
      description: "Comprehensive app documentation"
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'bg-emerald-500';
      case 'training': return 'bg-blue-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'ready': return 'Ready';
      case 'training': return 'Training';
      case 'error': return 'Error';
      default: return 'Idle';
    }
  };

  return (
    <div className="w-72 glass-effect border-r flex flex-col animate-fade-in-up">
      {/* Sidebar Header */}
      <div className="p-8 border-b border-border/50">
        <div className="flex items-center space-x-3 mb-2">
          <div className="w-12 h-12 rounded-xl flex items-center justify-center">
            <img 
              src={logoImage} 
              alt="Mastishk Logo" 
              className="w-full h-full object-contain"
            />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">
              <span className="font-extrabold text-primary">Mastishk</span>
              <span className="text-xs text-muted-foreground ml-1">©</span>
            </h1>
            <p className="text-xs text-muted-foreground font-medium">
              Transformer Studio
            </p>
          </div>
        </div>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 px-6 py-8">
        <div className="space-y-8">
          {/* Model Configuration */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider px-2">
              Model
            </h3>
            {navigationItems.slice(0, 4).map((item) => {
              const Icon = item.icon;
              const isActive = location === item.path;
              
              return (
                <Link key={item.path} href={item.path}>
                  <a className={cn(
                    "group flex items-center space-x-4 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-300",
                    isActive 
                      ? "premium-button text-white shadow-lg" 
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50 premium-card"
                  )}>
                    <Icon className="w-5 h-5 transition-transform group-hover:scale-110" />
                    <span>{item.label}</span>
                  </a>
                </Link>
              );
            })}
          </div>

          {/* Analysis & Visualization */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider px-2">
              Analysis
            </h3>
            {navigationItems.slice(4).map((item) => {
              const Icon = item.icon;
              const isActive = location === item.path;
              
              return (
                <Link key={item.path} href={item.path}>
                  <a className={cn(
                    "group flex items-center space-x-4 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-300",
                    isActive 
                      ? "premium-button text-white shadow-lg" 
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50 premium-card"
                  )}>
                    <Icon className="w-5 h-5 transition-transform group-hover:scale-110" />
                    <span>{item.label}</span>
                  </a>
                </Link>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Model Status */}
      <div className="p-6 border-t border-border/50">
        <div className="premium-card rounded-2xl p-5 content-container">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-bold text-foreground text-truncate">Model Status</span>
            <div className={cn(
              "w-3 h-3 rounded-full shadow-lg flex-shrink-0",
              modelStatus?.status === 'training' ? "bg-emerald-500 shadow-emerald-500/50 animate-pulse" :
              modelStatus?.status === 'ready' ? "bg-blue-500 shadow-blue-500/50" : "bg-slate-400"
            )} />
          </div>
          <p className="text-xs text-muted-foreground font-medium leading-relaxed break-words">
            {modelStatus?.status === 'training' ? 'Currently training model...' :
             modelStatus?.status === 'ready' ? `Ready to use • ${modelStatus?.lastTrained || 'Recently'}` :
             'No model configured yet'}
          </p>
        </div>
      </div>

      {/* Copyright */}
      <div className="p-4 border-t border-border/50 mt-auto">
        <div className="text-center">
          <p className="text-xs text-muted-foreground">
            © 2025 <span className="font-semibold">Aman Sharma</span>
          </p>
          <p className="text-xs text-muted-foreground/70 mt-1">
            All rights reserved
          </p>
        </div>
      </div>
    </div>
  );
}
