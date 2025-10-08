#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

async function buildStaticSite() {
    console.log('🔨 Building static site...');
    
    try {
        // Read the manifest
        const manifestPath = path.join(__dirname, 'manifest.json');
        const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
        
        // Read the base index.html
        const indexPath = path.join(__dirname, 'index.html');
        let indexContent = fs.readFileSync(indexPath, 'utf8');
        
        // Build the component content
        let componentsHtml = '';
        
        // Sort sections by order and load each component
        const sortedSections = manifest.sections.sort((a, b) => a.order - b.order);
        
        for (const section of sortedSections) {
            console.log(`📄 Loading section: ${section.title}`);
            
            const componentPath = path.join(__dirname, 'components', section.file);
            
            if (fs.existsSync(componentPath)) {
                const componentHtml = fs.readFileSync(componentPath, 'utf8');
                
                // Wrap component in section container (matching the dynamic loader)
                componentsHtml += `
                <div id="${section.id}" class="manuscript-section" data-section="${section.title}">
                    ${componentHtml}
                </div>`;
            } else {
                console.warn(`⚠️  Component not found: ${section.file}`);
                componentsHtml += `
                <div id="${section.id}" class="manuscript-section" data-section="${section.title}">
                    <div style="background-color: #ffe6e6; border: 1px solid #ff9999; padding: 15px; margin: 20px 0; border-radius: 5px;">
                        <h3 style="color: #cc0000; margin-top: 0;">Missing Section: ${section.title}</h3>
                        <p>Component file <code>components/${section.file}</code> not found</p>
                    </div>
                </div>`;
            }
        }
        
        // Replace the dynamic content with static content
        // Remove the loading div and script, insert the content directly
        const staticContent = indexContent
            // Replace the loading div and manuscript-content div with the built content
            .replace(
                /<div id="loading".*?<\/div>\s*<div id="manuscript-content".*?<\/div>/s,
                `<div id="manuscript-content">${componentsHtml}</div>`
            )
            // Remove the component loading script (keep other scripts)
            .replace(
                /<!-- Component Loading Script -->[\s\S]*?<\/script>/,
                '<!-- Static build - components pre-loaded -->'
            )
            // Add a comment indicating this is a built version
            .replace(
                '<main id="main-content" role="main">',
                '<!-- This is a statically built version for SEO/deployment -->\n    <main id="main-content" role="main">'
            );
        
        // Create dist directory if it doesn't exist
        const distDir = path.join(__dirname, 'dist');
        if (!fs.existsSync(distDir)) {
            fs.mkdirSync(distDir, { recursive: true });
        }
        
        // Write the built file
        const outputPath = path.join(distDir, 'index.html');
        fs.writeFileSync(outputPath, staticContent, 'utf8');
        
        // Copy necessary static assets to dist
        const assetDirs = ['public', 'figures', 'data'];
        for (const assetDir of assetDirs) {
            const srcPath = path.join(__dirname, assetDir);
            const destPath = path.join(distDir, assetDir);
            
            if (fs.existsSync(srcPath)) {
                console.log(`📁 Copying ${assetDir}/`);
                copyRecursiveSync(srcPath, destPath);
            }
        }
        
        // Copy manifest.json for reference
        fs.copyFileSync(manifestPath, path.join(distDir, 'manifest.json'));
        
        console.log('✅ Static site built successfully!');
        console.log(`📁 Output: ${outputPath}`);
        console.log(`📊 Generated ${manifest.sections.length} sections`);
        console.log('🚀 Ready for deployment');
        
    } catch (error) {
        console.error('❌ Build failed:', error);
        process.exit(1);
    }
}

// Helper function to copy directories recursively
function copyRecursiveSync(src, dest) {
    const exists = fs.existsSync(src);
    const stats = exists && fs.statSync(src);
    const isDirectory = exists && stats.isDirectory();
    
    if (isDirectory) {
        if (!fs.existsSync(dest)) {
            fs.mkdirSync(dest, { recursive: true });
        }
        fs.readdirSync(src).forEach(childItemName => {
            copyRecursiveSync(
                path.join(src, childItemName),
                path.join(dest, childItemName)
            );
        });
    } else {
        fs.copyFileSync(src, dest);
    }
}

// Run if called directly
if (require.main === module) {
    buildStaticSite();
}

module.exports = { buildStaticSite };
