const fs = require('fs');
const path = require('path');

// Function to recursively find all .jsx and .js files
function findFiles(dir, extensions = ['.jsx', '.js']) {
  let results = [];
  const list = fs.readdirSync(dir);
  
  list.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat && stat.isDirectory() && file !== 'node_modules') {
      results = results.concat(findFiles(filePath, extensions));
    } else if (extensions.some(ext => file.endsWith(ext))) {
      results.push(filePath);
    }
  });
  
  return results;
}

// Function to replace framer-motion imports and usage
function replaceFramerMotion(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');
  let modified = false;
  
  // Replace import statements
  if (content.includes('framer-motion')) {
    content = content.replace(/import\s*{[^}]*}\s*from\s*['"]framer-motion['"];?\s*/g, '');
    content = content.replace(/import\s+\*\s+as\s+\w+\s+from\s*['"]framer-motion['"];?\s*/g, '');
    modified = true;
  }
  
  // Replace motion components with regular divs
  content = content.replace(/motion\.(\w+)/g, '$1');
  content = content.replace(/<motion\.(\w+)/g, '<$1');
  content = content.replace(/<\/motion\.(\w+)/g, '</$1');
  
  // Replace animate props with CSS classes
  content = content.replace(/\s+animate\s*=\s*{[^}]*}/g, '');
  content = content.replace(/\s+initial\s*=\s*{[^}]*}/g, '');
  content = content.replace(/\s+exit\s*=\s*{[^}]*}/g, '');
  content = content.replace(/\s+transition\s*=\s*{[^}]*}/g, '');
  content = content.replace(/\s+whileHover\s*=\s*{[^}]*}/g, '');
  content = content.replace(/\s+whileTap\s*=\s*={[^}]*}/g, '');
  
  if (modified) {
    fs.writeFileSync(filePath, content);
    console.log(`Updated: ${filePath}`);
  }
}

// Main execution
const srcDir = path.join(__dirname, 'src');
const files = findFiles(srcDir);

console.log(`Found ${files.length} files to process...`);

files.forEach(file => {
  try {
    replaceFramerMotion(file);
  } catch (error) {
    console.error(`Error processing ${file}:`, error.message);
  }
});

console.log('Framer Motion replacement complete!');
