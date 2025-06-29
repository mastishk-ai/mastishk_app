import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { UploadCloud, File, X, Database, ExternalLink, AlertCircle, Globe, Link } from "lucide-react";
import { useTraining } from "@/hooks/use-training";
import { useToast } from "@/hooks/use-toast";

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
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [crawling, setCrawling] = useState(false);
  const [crawlUrl, setCrawlUrl] = useState('');
  const [crawlOptions, setCrawlOptions] = useState({
    includeSubdomains: false,
    maxPages: 10,
    extractText: true,
    extractLinks: false,
    respectRobots: true
  });
  const { uploadDataMutation, isUploading } = useTraining();
  const { toast } = useToast();

  const handleFiles = useCallback(async (files: FileList) => {
    setUploadError(null);
    
    try {
      // Preview first file if it's text (do this before upload to avoid blocking)
      if (files[0] && files[0].type.startsWith('text/')) {
        const text = await files[0].text();
        setPreviewText(text.slice(0, 500) + (text.length > 500 ? '...' : ''));
      }
      
      // Upload files using mutation
      uploadDataMutation.mutate(files, {
        onSuccess: (result) => {
          if (result && result.files) {
            setUploadedFiles(prev => [...prev, ...result.files]);
            toast({
              title: "Upload Successful",
              description: `${result.files.length} files uploaded successfully`,
            });
          }
        },
        onError: (error) => {
          const errorMessage = error instanceof Error ? error.message : 'Upload failed';
          setUploadError(errorMessage);
          toast({
            title: "Upload Failed",
            description: errorMessage,
            variant: "destructive",
          });
        }
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setUploadError(errorMessage);
      console.error('Upload failed:', error);
    }
  }, [uploadDataMutation, toast]);

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

  const crawlWebsite = async () => {
    if (!crawlUrl.trim()) {
      toast({
        title: "Error",
        description: "Please enter a valid URL to crawl",
        variant: "destructive"
      });
      return;
    }

    setCrawling(true);
    try {
      const response = await fetch('/api/training/crawl', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: crawlUrl,
          options: crawlOptions
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to crawl website');
      }

      const result = await response.json();
      
      // Add crawled data as a new file
      const crawledFile: UploadedFile = {
        filename: `crawled_${new URL(crawlUrl).hostname}_${Date.now()}.txt`,
        originalName: `crawled_${new URL(crawlUrl).hostname}.txt`,
        path: '/temp/crawled',
        size: result.content?.length || 0
      };

      setUploadedFiles(prev => [...prev, crawledFile]);
      setCrawlUrl('');
      
      toast({
        title: "Website Crawled Successfully",
        description: `Extracted content from ${result.pagesProcessed || 1} page(s)`,
      });
    } catch (error) {
      console.error('Error crawling website:', error);
      toast({
        title: "Crawl Failed",
        description: "Failed to crawl the website. Please check the URL and try again.",
        variant: "destructive"
      });
    } finally {
      setCrawling(false);
    }
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
        <p className="text-sm text-muted-foreground">Upload files, crawl websites, or use sample datasets</p>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* File Upload Section */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-foreground">File Upload</h4>
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
                onChange={(e) => e.target.files && handleFiles(e.target.files)}
                accept=".txt,.json,.jsonl,.csv"
              />
            </div>
          </div>

          {/* Web Crawling Section */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-foreground">Web Crawling</h4>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="crawl-url">Website URL</Label>
                <div className="flex space-x-2">
                  <Input
                    id="crawl-url"
                    type="url"
                    placeholder="https://example.com"
                    value={crawlUrl}
                    onChange={(e) => setCrawlUrl(e.target.value)}
                    className="flex-1"
                  />
                  <Button 
                    onClick={crawlWebsite} 
                    disabled={crawling || !crawlUrl.trim()}
                    className="flex items-center gap-2"
                  >
                    {crawling ? (
                      <>
                        <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                        Crawling...
                      </>
                    ) : (
                      <>
                        <Link className="w-4 h-4" />
                        Crawl Site
                      </>
                    )}
                  </Button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 border rounded-lg bg-muted/50">
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <div className="flex flex-col items-center space-y-1">
                      <Switch
                        checked={crawlOptions.includeSubdomains}
                        onCheckedChange={(checked) => 
                          setCrawlOptions(prev => ({ ...prev, includeSubdomains: checked }))
                        }
                      />
                      <span className={`text-xs font-medium ${crawlOptions.includeSubdomains ? 'text-green-600' : 'text-gray-500'}`}>
                        {crawlOptions.includeSubdomains ? 'ON' : 'OFF'}
                      </span>
                    </div>
                    <div className="flex-1">
                      <Label className="text-sm font-medium">Include Subdomains</Label>
                      <p className="text-xs text-muted-foreground">Crawl subdomains of the target site</p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <div className="flex flex-col items-center space-y-1">
                      <Switch
                        checked={crawlOptions.extractText}
                        onCheckedChange={(checked) => 
                          setCrawlOptions(prev => ({ ...prev, extractText: checked }))
                        }
                      />
                      <span className={`text-xs font-medium ${crawlOptions.extractText ? 'text-green-600' : 'text-gray-500'}`}>
                        {crawlOptions.extractText ? 'ON' : 'OFF'}
                      </span>
                    </div>
                    <div className="flex-1">
                      <Label className="text-sm font-medium">Extract Text Content</Label>
                      <p className="text-xs text-muted-foreground">Extract clean text for training</p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <div className="flex flex-col items-center space-y-1">
                      <Switch
                        checked={crawlOptions.respectRobots}
                        onCheckedChange={(checked) => 
                          setCrawlOptions(prev => ({ ...prev, respectRobots: checked }))
                        }
                      />
                      <span className={`text-xs font-medium ${crawlOptions.respectRobots ? 'text-green-600' : 'text-gray-500'}`}>
                        {crawlOptions.respectRobots ? 'ON' : 'OFF'}
                      </span>
                    </div>
                    <div className="flex-1">
                      <Label className="text-sm font-medium">Respect robots.txt</Label>
                      <p className="text-xs text-muted-foreground">Follow site crawling rules</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="space-y-1">
                    <Label className="text-sm font-medium">Max Pages to Crawl</Label>
                    <Input
                      type="number"
                      min="1"
                      max="100"
                      value={crawlOptions.maxPages}
                      onChange={(e) => 
                        setCrawlOptions(prev => ({ ...prev, maxPages: parseInt(e.target.value) || 10 }))
                      }
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Sample Data Section */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-foreground">Sample Data Sources</h4>
            <div className="space-y-2">
              <Button variant="outline" className="w-full justify-between">
                <span>Hugging Face Datasets</span>
                <ExternalLink className="w-4 h-4" />
              </Button>
              <Button variant="outline" className="w-full justify-between">
                <span>Sample Text Datasets</span>
                <Database className="w-4 h-4" />
              </Button>
            </div>
          </div>
          
          <TabsContent value="upload" className="mt-6">
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
              <div className="space-y-2 bg-background p-4 rounded-lg border">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">Uploading files...</span>
                  <span className="text-sm text-muted-foreground">Please wait</span>
                </div>
                <Progress value={75} className="w-full" />
                <p className="text-xs text-muted-foreground">
                  Files are being processed and saved to the server
                </p>
              </div>
            )}

            {uploadError && (
              <div className="space-y-2 bg-destructive/10 p-4 rounded-lg border border-destructive/20">
                <div className="flex items-center space-x-2">
                  <AlertCircle className="w-4 h-4 text-destructive" />
                  <span className="text-sm font-medium text-destructive">Upload Failed</span>
                </div>
                <p className="text-xs text-muted-foreground">{uploadError}</p>
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
        </Tabs>
      </CardContent>
    </Card>
  );
}
