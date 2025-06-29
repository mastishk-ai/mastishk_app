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
  Box,
  Eye,
  FileText,
  Shield,
  Heart,
  Mail,
  ChevronUp,
  ChevronDown
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
  const [isModelStatusCollapsed, setIsModelStatusCollapsed] = useState(false);
  const [isLegalCollapsed, setIsLegalCollapsed] = useState(true);

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
      path: "/visualization",
      label: "3D Visualization",
      icon: Eye,
      description: "Interactive 3D model architecture"
    },
    {
      path: "/docs",
      label: "Documentation",
      icon: Box,
      description: "Comprehensive app documentation"
    }
  ];

  const legalItems = [
    {
      path: "/about",
      label: "About",
      icon: Heart,
      description: "About Mastishk Transformer Studio"
    },
    {
      path: "/terms",
      label: "Terms of Service",
      icon: FileText,
      description: "Terms and conditions"
    },
    {
      path: "/privacy",
      label: "Privacy Policy", 
      icon: Shield,
      description: "Privacy policy and data handling"
    },
    {
      path: "/contact",
      label: "Contact Us",
      icon: Mail,
      description: "Get in touch with our team"
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
      <nav className="flex-1 px-6 py-8 overflow-y-auto">
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

          {/* Legal & Info */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider px-2">
              Legal & Info
            </h3>
            {legalItems.map((item) => {
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
      <div className="border-t border-border/50">
        <div className="p-4">
          <button 
            onClick={() => setIsModelStatusCollapsed(!isModelStatusCollapsed)}
            className="w-full flex items-center justify-between p-2 hover:bg-muted/50 rounded-lg transition-colors"
          >
            <span className="text-sm font-bold text-foreground">Model Status</span>
            <div className="flex items-center space-x-2">
              <div className={cn(
                "w-3 h-3 rounded-full shadow-lg flex-shrink-0",
                modelStatus?.status === 'training' ? "bg-emerald-500 shadow-emerald-500/50 animate-pulse" :
                modelStatus?.status === 'ready' ? "bg-blue-500 shadow-blue-500/50" : "bg-slate-400"
              )} />
              {isModelStatusCollapsed ? 
                <ChevronDown className="w-4 h-4 text-muted-foreground" /> : 
                <ChevronUp className="w-4 h-4 text-muted-foreground" />
              }
            </div>
          </button>
          
          {!isModelStatusCollapsed && (
            <div className="mt-2 p-3 premium-card rounded-xl">
              <p className="text-xs text-muted-foreground font-medium leading-relaxed break-words">
                {modelStatus?.status === 'training' ? 'Currently training model...' :
                 modelStatus?.status === 'ready' ? `Ready to use • ${modelStatus?.lastTrained || 'Recently'}` :
                 'No model configured yet'}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Legal & Info */}
      <div className="border-t border-border/50 mt-auto">
        <div className="p-4">
          <button 
            onClick={() => setIsLegalCollapsed(!isLegalCollapsed)}
            className="w-full flex items-center justify-between p-2 hover:bg-muted/50 rounded-lg transition-colors"
          >
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-bold text-foreground">Legal & Info</span>
            </div>
            {isLegalCollapsed ? 
              <ChevronDown className="w-4 h-4 text-muted-foreground" /> : 
              <ChevronUp className="w-4 h-4 text-muted-foreground" />
            }
          </button>
          
          {!isLegalCollapsed && (
            <div className="mt-2 space-y-2">
              <Link href="/about" className={cn(
                "flex items-center space-x-3 px-3 py-2 rounded-xl text-sm font-medium transition-all duration-300 hover:scale-[1.02] group",
                location === "/about" 
                  ? "bg-gradient-to-r from-primary/20 to-primary/10 text-primary shadow-lg shadow-primary/20 border border-primary/30" 
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/60"
              )}>
                <Heart className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">About</span>
              </Link>
              
              <Link href="/terms" className={cn(
                "flex items-center space-x-3 px-3 py-2 rounded-xl text-sm font-medium transition-all duration-300 hover:scale-[1.02] group",
                location === "/terms" 
                  ? "bg-gradient-to-r from-primary/20 to-primary/10 text-primary shadow-lg shadow-primary/20 border border-primary/30" 
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/60"
              )}>
                <FileText className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">Terms of Service</span>
              </Link>
              
              <Link href="/privacy" className={cn(
                "flex items-center space-x-3 px-3 py-2 rounded-xl text-sm font-medium transition-all duration-300 hover:scale-[1.02] group",
                location === "/privacy" 
                  ? "bg-gradient-to-r from-primary/20 to-primary/10 text-primary shadow-lg shadow-primary/20 border border-primary/30" 
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/60"
              )}>
                <Shield className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">Privacy Policy</span>
              </Link>
              
              <Link href="/contact" className={cn(
                "flex items-center space-x-3 px-3 py-2 rounded-xl text-sm font-medium transition-all duration-300 hover:scale-[1.02] group",
                location === "/contact" 
                  ? "bg-gradient-to-r from-primary/20 to-primary/10 text-primary shadow-lg shadow-primary/20 border border-primary/30" 
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/60"
              )}>
                <Mail className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">Contact Us</span>
              </Link>
            </div>
          )}
        </div>
        
        {/* Copyright */}
        <div className="text-center p-4 border-t border-border/50">
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
