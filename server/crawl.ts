import express from 'express';
import { z } from 'zod';
import FirecrawlApp from '@mendable/firecrawl-js';

const router = express.Router();

// Crawl options schema
const CrawlOptionsSchema = z.object({
  url: z.string().url(),
  options: z.object({
    includeSubdomains: z.boolean().default(false),
    maxPages: z.number().min(1).max(100).default(10),
    extractText: z.boolean().default(true),
    extractLinks: z.boolean().default(false),
    respectRobots: z.boolean().default(true)
  })
});

router.post('/crawl', async (req, res) => {
  try {
    const { url, options } = CrawlOptionsSchema.parse(req.body);
    
    // For now, return a mock response since we need API key setup
    // In production, this would use: const app = new FirecrawlApp({apiKey: process.env.FIRECRAWL_API_KEY});
    
    const mockContent = `
# Sample Content from ${new URL(url).hostname}

This is mock content extracted from the website. In a real implementation, 
this would contain the actual text content from the crawled pages.

The crawler would respect the following settings:
- Include subdomains: ${options.includeSubdomains ? 'Yes' : 'No'}
- Max pages: ${options.maxPages}
- Extract text: ${options.extractText ? 'Yes' : 'No'}
- Respect robots.txt: ${options.respectRobots ? 'Yes' : 'No'}

This content would be processed and cleaned for training purposes.
    `.trim();

    res.json({
      success: true,
      content: mockContent,
      pagesProcessed: Math.min(options.maxPages, 3),
      url: url,
      options: options
    });

  } catch (error) {
    console.error('Crawl error:', error);
    res.status(400).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to crawl website'
    });
  }
});

export default router;