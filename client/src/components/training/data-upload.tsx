import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { UploadCloud, File, X, Database, ExternalLink } from "lucide-react";
import { useTraining } from "@/hooks/use-training";

interface UploadedFile {
  filename: string;
  originalName: string;
  path: string;
  size: number;
}

export function DataUpload() {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [previewText, setPreviewText] = useState("");
  const { uploadData, isUploading } = useTraining();

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  }, []);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  }, []);

  const handleFiles = async (files: FileList) => {
    try {
      const result = await uploadData(files);
      setUploadedFiles(prev => [...prev, ...result.files]);
      
      // Preview first file if it's text
      if (files[0] && files[0].type.startsWith('text/')) {
        const text = await files[0].text();
        setPreviewText(text.slice(0, 500) + (text.length > 500 ? '...' : ''));
      }
    } catch (error) {
      console.error('Upload failed:', error);
    }
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const totalSize = uploadedFiles.reduce((sum, file) => sum + file.size, 0);
  const avgLength = uploadedFiles.length > 0 ? "256 tokens" : "0 tokens"; // This would be calculated from actual content

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50">
        <CardTitle className="flex items-center">
          <UploadCloud className="w-5 h-5 mr-2 text-blue-500" />
          Training Data
        </CardTitle>
        <p className="text-sm text-muted-foreground">Upload and manage your training datasets</p>
      </CardHeader>
      <CardContent className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* File Upload Area */}
          <div className="space-y-4">
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
                dragActive 
                  ? 'border-primary bg-primary/5' 
                  : 'border-muted-foreground hover:border-muted-foreground/70'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-upload')?.click()}
            >
              <UploadCloud className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-foreground font-medium mb-2">Drop files here or click to upload</p>
              <p className="text-sm text-muted-foreground">Supports .txt, .json, .jsonl, .csv</p>
              <input
                id="file-upload"
                type="file"
                className="hidden"
                multiple
                accept=".txt,.json,.jsonl,.csv"
                onChange={handleChange}
              />
            </div>

            {isUploading && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Uploading...</span>
                  <span className="text-sm text-muted-foreground">Processing files</span>
                </div>
                <Progress value={65} className="w-full" />
              </div>
            )}

            {/* Uploaded Files List */}
            {uploadedFiles.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-foreground">Uploaded Files</h4>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {uploadedFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-muted rounded-lg">
                      <div className="flex items-center space-x-2">
                        <File className="w-4 h-4 text-muted-foreground" />
                        <div>
                          <p className="text-sm font-medium text-foreground">{file.originalName}</p>
                          <p className="text-xs text-muted-foreground">{formatFileSize(file.size)}</p>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(index)}
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Data Source Options */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-foreground">Quick Sources</h4>
              <Button variant="outline" className="w-full justify-between">
                <span>Hugging Face Datasets</span>
                <ExternalLink className="w-4 h-4" />
              </Button>
              <Button variant="outline" className="w-full justify-between">
                <span>Sample Datasets</span>
                <Database className="w-4 h-4" />
              </Button>
            </div>
          </div>
          
          {/* Dataset Preview and Stats */}
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-semibold text-foreground mb-2">Dataset Preview</h4>
              <div className="bg-muted rounded-lg border p-4 max-h-64 overflow-y-auto">
                {previewText ? (
                  <div className="text-sm text-foreground whitespace-pre-wrap font-mono">
                    {previewText}
                  </div>
                ) : uploadedFiles.length > 0 ? (
                  <div className="text-sm text-muted-foreground">
                    Preview will appear here after uploading text files...
                  </div>
                ) : (
                  <div className="text-sm text-muted-foreground">
                    Upload files to see preview
                  </div>
                )}
              </div>
            </div>
            
            {/* Dataset Stats */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-muted rounded-lg p-3 border">
                <div className="text-sm text-muted-foreground">Total Files</div>
                <div className="text-lg font-semibold text-foreground">{uploadedFiles.length}</div>
              </div>
              <div className="bg-muted rounded-lg p-3 border">
                <div className="text-sm text-muted-foreground">Total Size</div>
                <div className="text-lg font-semibold text-foreground">{formatFileSize(totalSize)}</div>
              </div>
            </div>

            {uploadedFiles.length > 0 && (
              <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span className="text-sm font-medium text-blue-400">Ready for Training</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {uploadedFiles.length} files uploaded and processed
                </p>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
