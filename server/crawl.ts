import { Router } from 'express';

const router = Router();

interface CrawlOptions {
  includeSubdomains: boolean;
  maxPages: number;
  extractText: boolean;
  extractLinks: boolean;
  respectRobots: boolean;
}

interface CrawlRequest {
  url: string;
  options: CrawlOptions;
}

// POST /api/training/crawl - Crawl a website for training data
router.post('/crawl', async (req, res) => {
  try {
    const { url, options }: CrawlRequest = req.body;

    if (!url) {
      return res.status(400).json({ error: 'URL is required' });
    }

    // Validate URL format
    try {
      new URL(url);
    } catch {
      return res.status(400).json({ error: 'Invalid URL format' });
    }

    console.log(`Starting crawl for ${url} with options:`, options);

    // For now, return a mock response until Firecrawl is properly configured
    const mockContent = `Sample content from ${url}\n\nThis is placeholder content that would be replaced with actual crawled content when Firecrawl API key is configured.\n\nThe content would include the main text from the webpage, extracted and cleaned for training purposes.`;

    console.log(`Mock crawl completed for: ${url}`);

    res.json({
      success: true,
      content: mockContent,
      pagesProcessed: 1,
      contentLength: mockContent.length,
      url: url,
      note: 'This is mock data. Set FIRECRAWL_API_KEY environment variable to enable real crawling.'
    });

  } catch (error) {
    console.error('Crawl error:', error);
    
    res.status(500).json({ 
      error: 'Internal server error during crawl',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;