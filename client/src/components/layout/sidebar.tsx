import { useState } from "react";
import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Settings, 
  Play, 
  Edit3, 
  Activity, 
  Save, 
  BarChart2, 
  Cpu,
  Zap
} from "lucide-react";

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
      label: "Text Generation",
      icon: Edit3,
      description: "Generate text with your model"
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
    <div className="w-64 bg-sidebar-background border-r border-sidebar-border flex flex-col h-full">
      {/* Logo & Title */}
      <div className="p-6 border-b border-sidebar-border">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-sidebar-primary-foreground">Mastishk Studio</h1>
            <p className="text-xs text-sidebar-foreground">Advanced LLM Platform</p>
          </div>
        </div>
      </div>
      
      {/* Navigation Menu */}
      <nav className="flex-1 p-4 space-y-2">
        <div className="space-y-1">
          {navigationItems.map((item) => {
            const isActive = location === item.path;
            return (
              <Link key={item.path} href={item.path}>
                <button className={`sidebar-nav-item w-full ${isActive ? 'active' : ''}`}>
                  <item.icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{item.label}</span>
                </button>
              </Link>
            );
          })}
        </div>
        
        {/* Model Status */}
        <div className="mt-8 p-3 bg-sidebar-primary rounded-lg border border-sidebar-border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-sidebar-foreground">Model Status</span>
            <div className={`w-2 h-2 rounded-full ${getStatusColor(modelStatus?.status || 'idle')}`}></div>
          </div>
          <p className="text-sm text-sidebar-primary-foreground">{getStatusText(modelStatus?.status || 'idle')}</p>
          {modelStatus?.lastTrained && (
            <p className="text-xs text-sidebar-foreground mt-1">
              Last trained: {modelStatus.lastTrained}
            </p>
          )}
        </div>
      </nav>
      
      {/* Quick Actions */}
      <div className="p-4 border-t border-sidebar-border">
        <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white" size="sm">
          <Zap className="w-4 h-4 mr-2" />
          Quick Train
        </Button>
      </div>
    </div>
  );
}
