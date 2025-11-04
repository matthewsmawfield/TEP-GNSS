#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * HTML to Markdown Converter for TEP-GNSS Site
 * Converts the built static HTML site into a clean markdown document
 */

class HTMLToMarkdownConverter {
    constructor() {
        this.output = '';
        this.currentSection = '';
    }

    /**
     * Convert HTML string to markdown with proper academic formatting
     */
    htmlToMarkdown(html) {
        // Remove script tags and their content
        html = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
        
        // Remove style tags and their content
        html = html.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '');
        
        // Remove comments
        html = html.replace(/<!--[\s\S]*?-->/g, '');
        
        // Preserve MathJax expressions before processing
        const mathExpressions = [];
        html = html.replace(/<span[^>]*class=["'][^"']*MathJax[^"']*["'][^>]*>.*?<\/span>/gi, (match) => {
            mathExpressions.push(match);
            return `__MATH_EXPRESSION_${mathExpressions.length - 1}__`;
        });
        
        // Convert manuscript sections to proper markdown structure FIRST
        html = html.replace(/<div[^>]*class=["'][^"']*manuscript-section[^"']*["'][^>]*data-section=["']([^"']*)["'][^>]*>/gi, '\n\n## $1\n\n');
        
        // Convert headers
        html = html.replace(/<h1[^>]*>(.*?)<\/h1>/gis, '\n# $1\n\n');
        html = html.replace(/<h2[^>]*>(.*?)<\/h2>/gis, '\n## $1\n\n');
        html = html.replace(/<h3[^>]*>(.*?)<\/h3>/gis, '\n### $1\n\n');
        html = html.replace(/<h4[^>]*>(.*?)<\/h4>/gis, '\n#### $1\n\n');
        
        // Convert paragraphs - use 's' flag to match across newlines
        html = html.replace(/<p[^>]*>(.*?)<\/p>/gis, '$1\n\n');
        
        // Convert strong/bold
        html = html.replace(/<(strong|b)[^>]*>(.*?)<\/(strong|b)>/gi, '**$2**');
        
        // Convert emphasis/italic
        html = html.replace(/<(em|i)[^>]*>(.*?)<\/(em|i)>/gi, '*$2*');
        
        // Convert links
        html = html.replace(/<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)<\/a>/gi, '[$2]($1)');
        
        // Convert lists
        html = html.replace(/<ul[^>]*>/gi, '\n');
        html = html.replace(/<\/ul>/gi, '\n');
        html = html.replace(/<ol[^>]*>/gi, '\n');
        html = html.replace(/<\/ol>/gi, '\n');
        html = html.replace(/<li[^>]*>(.*?)<\/li>/gi, '- $1\n');
        
        // Convert blockquotes
        html = html.replace(/<blockquote[^>]*>(.*?)<\/blockquote>/gi, '\n> $1\n\n');
        
        // Convert code blocks
        html = html.replace(/<pre[^>]*><code[^>]*>(.*?)<\/code><\/pre>/gi, '\n```\n$1\n```\n\n');
        html = html.replace(/<code[^>]*>(.*?)<\/code>/gi, '`$1`');
        
        // Convert line breaks
        html = html.replace(/<br\s*\/?>/gi, '\n');
        
        // Convert horizontal rules
        html = html.replace(/<hr\s*\/?>/gi, '\n---\n\n');
        
        // Convert divs with special classes to markdown equivalents
        html = html.replace(/<div[^>]*class=["'][^"']*abstract[^"']*["'][^>]*>/gi, '');
        html = html.replace(/<div[^>]*class=["'][^"']*theorem[^"']*["'][^>]*>/gi, '\n**Theorem:**\n');
        html = html.replace(/<div[^>]*class=["'][^"']*principle[^"']*["'][^>]*>/gi, '\n**Principle:**\n');
        html = html.replace(/<div[^>]*class=["'][^"']*proof[^"']*["'][^>]*>/gi, '\n*Proof:*\n');
        html = html.replace(/<div[^>]*class=["'][^"']*experimental-section[^"']*["'][^>]*>/gi, '\n**Experimental Section:**\n');
        html = html.replace(/<div[^>]*class=["'][^"']*critical-analysis[^"']*["'][^>]*>/gi, '\n**Critical Analysis:**\n');
        html = html.replace(/<div[^>]*class=["'][^"']*significance[^"']*["'][^>]*>/gi, '\n**Significance:**\n');
        
        // Handle abstract section specially - remove the h2 Abstract header since we already have the section header
        html = html.replace(/## Abstract\s*\n\s*<h2>Abstract<\/h2>/gi, '## Abstract\n\n');
        
        // Convert tables
        html = html.replace(/<table[^>]*>(.*?)<\/table>/gis, (match) => {
            return this.convertTable(match);
        });
        
        // Decode HTML entities BEFORE removing tags to prevent < and > from being interpreted as tag delimiters
        html = html.replace(/&amp;/g, '&');
        html = html.replace(/&lt;/g, '<');
        html = html.replace(/&gt;/g, '>');
        html = html.replace(/&quot;/g, '"');
        html = html.replace(/&#39;/g, "'");
        html = html.replace(/&nbsp;/g, ' ');
        html = html.replace(/&times;/g, '×');
        html = html.replace(/&minus;/g, '−');
        html = html.replace(/&plusmn;/g, '±');
        html = html.replace(/&sup2;/g, '²');
        html = html.replace(/&sup3;/g, '³');
        html = html.replace(/&sup1;/g, '¹');
        html = html.replace(/&deg;/g, '°');
        
        // Convert superscripts and subscripts BEFORE removing HTML tags
        // This preserves scientific notation like 10⁻¹⁰
        html = html.replace(/<sup[^>]*>(.*?)<\/sup>/gi, (match, content) => {
            // Convert digits to superscript Unicode
            const superscriptMap = {
                '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                '-': '⁻', '−': '⁻', '+': '⁺', '=': '⁼', '(': '⁽', ')': '⁾'
            };
            return content.split('').map(c => superscriptMap[c] || c).join('');
        });
        
        html = html.replace(/<sub[^>]*>(.*?)<\/sub>/gi, (match, content) => {
            // Convert digits to subscript Unicode
            const subscriptMap = {
                '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
                '-': '₋', '−': '₋', '+': '₊', '=': '₌', '(': '₍', ')': '₎'
            };
            return content.split('').map(c => subscriptMap[c] || c).join('');
        });
        
        // Remove remaining HTML tags (but not < or > used as comparison operators)
        // Only match tags that start with < followed by a letter or /
        html = html.replace(/<\/?[a-zA-Z][^>]*>/g, '');
        
        // Restore MathJax expressions
        mathExpressions.forEach((expr, index) => {
            html = html.replace(`__MATH_EXPRESSION_${index}__`, expr);
        });
        
        // Clean up whitespace
        html = html.replace(/\n\s*\n\s*\n/g, '\n\n');
        html = html.replace(/^\s+|\s+$/g, '');
        
        // Remove duplicate headers (same header appearing consecutively)
        html = html.replace(/(##\s+[^\n]+)\n+\1/g, '$1');
        
        // Clean up any remaining formatting issues
        html = html.replace(/\n{3,}/g, '\n\n');
        
        return html;
    }

    /**
     * Convert HTML table to markdown table
     */
    convertTable(tableHtml) {
        const rows = [];
        const rowMatches = tableHtml.match(/<tr[^>]*>(.*?)<\/tr>/gis);
        
        if (!rowMatches) return '';
        
        rowMatches.forEach((row, index) => {
            const cells = row.match(/<t[dh][^>]*>(.*?)<\/t[dh]>/gis);
            if (cells) {
                const cellTexts = cells.map(cell => 
                    cell.replace(/<[^>]+>/g, '').trim()
                );
                rows.push(cellTexts);
            }
        });
        
        if (rows.length === 0) return '';
        
        // Create markdown table
        let markdown = '\n';
        rows.forEach((row, index) => {
            markdown += '| ' + row.join(' | ') + ' |\n';
            if (index === 0) {
                // Add separator row
                markdown += '|' + row.map(() => ' --- ').join('|') + '|\n';
            }
        });
        markdown += '\n';
        
        return markdown;
    }

    /**
     * Extract title and metadata from HTML
     */
    extractMetadata(html) {
        const titleMatch = html.match(/<title[^>]*>(.*?)<\/title>/i);
        const title = titleMatch ? titleMatch[1] : 'Global Time Echoes: Distance-Structured Correlations in GNSS Clocks';
        
        const authorMatch = html.match(/<meta[^>]*name=["']author["'][^>]*content=["']([^"']*)["']/i);
        const author = authorMatch ? authorMatch[1] : 'Matthew Lukin Smawfield';
        
        const versionMatch = html.match(/<div[^>]*class=["'][^"']*version[^"']*["'][^>]*>(.*?)<\/div>/i);
        const version = versionMatch ? versionMatch[1]
            .replace(/<[^>]+>/g, '')
            .replace(/^Version:\s*/i, '')
            .trim() : 'v0.18 (Jaipur)';
        
        const dateMatch = html.match(/<div[^>]*class=["'][^"']*date[^"']*["'][^>]*>(.*?)<\/div>/i);
        const date = dateMatch ? dateMatch[1].replace(/<[^>]+>/g, '').trim() : 'First published: 17 September 2025 · Last updated: 13 October 2025';
        
        const doiMatch = html.match(/DOI:\s*<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)<\/a>/i);
        const doi = doiMatch ? doiMatch[2] : '10.5281/zenodo.17517142';
        
        return { title, author, version, date, doi };
    }

    /**
     * Extract main content from HTML - FIXED VERSION
     */
    extractMainContent(html) {
        // Find the manuscript-content div and extract everything until the closing main tag
        const startMatch = html.match(/<div[^>]*id=["']manuscript-content["'][^>]*>/i);
        if (!startMatch) {
            throw new Error('Could not find manuscript-content div');
        }
        
        const startIndex = startMatch.index + startMatch[0].length;
        
        // Find the closing main tag
        const endMatch = html.match(/<\/main>/i);
        if (!endMatch) {
            throw new Error('Could not find closing main tag');
        }
        
        const endIndex = endMatch.index;
        
        // Extract the content between these points
        const content = html.substring(startIndex, endIndex);
        
        return content;
    }

    /**
     * Convert the built HTML site to markdown
     */
    async convertSiteToMarkdown() {
        console.log('🔄 Converting HTML site to markdown...');
        
        try {
            // Read the built HTML file
            const htmlPath = path.join(__dirname, 'dist', 'index.html');
            if (!fs.existsSync(htmlPath)) {
                throw new Error('Built HTML file not found. Please run "npm run build" first.');
            }
            
            const html = fs.readFileSync(htmlPath, 'utf8');
            
            // Extract metadata
            const metadata = this.extractMetadata(html);
            
            // Extract main content
            const mainContent = this.extractMainContent(html);
            
            // Convert to markdown
            const markdownContent = this.htmlToMarkdown(mainContent);
            
            // Build the complete markdown document
            const markdown = this.buildMarkdownDocument(metadata, markdownContent);
            
            // Write to file (unique name for Paper 2)
            const outputPath = path.join(__dirname, '..', 'manuscript-code-longspan.md');
            fs.writeFileSync(outputPath, markdown, 'utf8');
            
            console.log('✅ Markdown conversion complete!');
            console.log(`📄 Output: ${outputPath}`);
            console.log(`📊 Document: ${metadata.title}`);
            console.log(`👤 Author: ${metadata.author}`);
            console.log(`📅 Version: ${metadata.version}`);
            
            return outputPath;
            
        } catch (error) {
            console.error('❌ Markdown conversion failed:', error.message);
            process.exit(1);
        }
    }

    /**
     * Build the complete markdown document with metadata
     */
    buildMarkdownDocument(metadata, content) {
        const timestamp = new Date().toISOString().split('T')[0];
        
        // Clean up the title to remove the author part
        const cleanTitle = metadata.title.replace(' | Matthew Lukin Smawfield', '');
        
        return `# ${cleanTitle}

**Author:** ${metadata.author}  
**Version:** ${metadata.version}  
**Date:** ${metadata.date}  
**DOI:** ${metadata.doi}  
**Generated:** ${timestamp}  

---

${content}

---

*This document was automatically generated from the TEP-GNSS research site. For the interactive version with figures and enhanced formatting, visit: https://matthewsmawfield.github.io/TEP-GNSS/*

*Source code and data available at: https://github.com/matthewsmawfield/TEP-GNSS*
`;
    }
}

// Main execution
async function main() {
    const converter = new HTMLToMarkdownConverter();
    await converter.convertSiteToMarkdown();
}

// Run if called directly
if (require.main === module) {
    main();
}

module.exports = { HTMLToMarkdownConverter };