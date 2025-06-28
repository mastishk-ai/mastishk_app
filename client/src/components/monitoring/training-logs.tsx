import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Terminal, Download, Trash2 } from "lucide-react";
import { useWebSocket } from "@/hooks/use-websocket";

interface LogEntry {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'success';
  message: string;
}

export function TrainingLogs() {
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      timestamp: new Date().toISOString(),
      level: 'info',
      message: 'Training system initialized and ready'
    }
  ]);
  
  const { lastMessage } = useWebSocket();

  useEffect(() => {
    if (lastMessage) {
      const timestamp = new Date().toISOString();
      let logEntry: LogEntry | null = null;

      switch (lastMessage.type) {
        case 'training_progress':
          const progress = lastMessage.data.progress;
          logEntry = {
            timestamp,
            level: 'info',
            message: `Training step ${progress.step}/10000 completed - Loss: ${progress.loss?.toFixed(4)} - LR: ${progress.learningRate?.toExponential(2)}`
          };
          break;
        
        case 'training_complete':
          logEntry = {
            timestamp,
            level: 'success',
            message: `Training completed successfully at step ${lastMessage.data.final_step}`
          };
          break;
        
        case 'training_error':
          logEntry = {
            timestamp,
            level: 'error',
            message: `Training error: ${lastMessage.data.error}`
          };
          break;
      }

      if (logEntry) {
        setLogs(prev => [...prev.slice(-99), logEntry!]); // Keep last 100 logs
      }
    }
  }, [lastMessage]);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'error': return '❌';
      case 'warning': return '⚠️';
      case 'success': return '✅';
      case 'info': 
      default: return 'ℹ️';
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'error': return 'text-red-400';
      case 'warning': return 'text-yellow-400';
      case 'success': return 'text-green-400';
      case 'info':
      default: return 'text-blue-400';
    }
  };

  const exportLogs = () => {
    const logText = logs
      .map(log => `[${formatTimestamp(log.timestamp)}] ${log.level.toUpperCase()}: ${log.message}`)
      .join('\n');
    
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_logs_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const clearLogs = () => {
    setLogs([]);
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <Terminal className="w-5 h-5 mr-2 text-slate-400" />
            Training Logs
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm" onClick={exportLogs}>
              <Download className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={clearLogs}>
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="bg-slate-950 p-4 h-64 overflow-y-auto font-mono text-sm">
          {logs.length > 0 ? (
            <div className="space-y-1">
              {logs.map((log, index) => (
                <div key={index} className="flex items-start space-x-2">
                  <span className="text-slate-500 text-xs shrink-0">
                    [{formatTimestamp(log.timestamp)}]
                  </span>
                  <span className="text-xs">{getLevelIcon(log.level)}</span>
                  <span className={`text-xs ${getLevelColor(log.level)} break-all`}>
                    {log.message}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-slate-500 text-center py-8">
              No logs available
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
