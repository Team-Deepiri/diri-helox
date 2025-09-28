import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Enhanced no-cache middleware for real-time updates
const noCacheMiddleware = () => ({
  name: 'no-cache',
  configureServer(server) {
    server.middlewares.use((_, res, next) => {
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.setHeader('Pragma', 'no-cache');
      res.setHeader('Expires', '0');
      next();
    });
  }
});

export default defineConfig({
  plugins: [
    react({
      // Enable fast refresh for better real-time updates
      fastRefresh: true,
      // Include HMR boundary for better component updates
      include: "**/*.{jsx,tsx}",
    }), 
    noCacheMiddleware()
  ],
  
  // Disable all caching for real-time development
  cacheDir: false,
  
  server: {
    host: '0.0.0.0', // Needed for Docker container access
    port: 5173,
    strictPort: true,
    
    // Enhanced file watching for real-time updates
    watch: {
      usePolling: true,
      interval: 100, // Check for changes every 100ms
      binaryInterval: 300,
      ignored: ['**/node_modules/**', '**/.git/**'], // Only ignore what's necessary
    },
    
    // Enhanced HMR configuration
    hmr: {
      port: 5173,
      clientPort: 5173,
      overlay: true, // Show errors in overlay for immediate feedback
    },
    
    // Proxy API calls to backend
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      },
      '/socket.io': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        ws: true, // Enable WebSocket proxying for Socket.IO
      }
    }
  },
  
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1000,
    // Enable source maps for better debugging
    sourcemap: process.env.NODE_ENV === 'development',
  },
  
  // Enhanced dependency optimization
  optimizeDeps: {
    include: ['react', 'react-dom', 'socket.io-client'],
    esbuildOptions: {
      loader: {
        '.js': 'jsx'
      }
    },
    // Force deps to be bundled to avoid issues
    force: false
  },
  
  // Enable CSS HMR
  css: {
    devSourcemap: true
  }
})



// import { defineConfig } from 'vite';
// import react from '@vitejs/plugin-react';

// export default defineConfig({
//   plugins: [
//     react({
//       jsxRuntime: 'classic', // Explicitly use classic JSX runtime
//       babel: {
//         plugins: [
//           ['@babel/plugin-transform-react-jsx', {
//             runtime: 'classic'
//           }]
//         ]
//       }
//     })
//   ],
//   server: {
//     port: 5173,
//     hmr: {
//       overlay: true // Disable HMR error overlay // was false before
//     }
//   },
//   build: {
//     outDir: 'dist',
//     emptyOutDir: true
//   },
//   optimizeDeps: {
//     esbuildOptions: {
//       loader: {
//         '.js': 'jsx' // Enable JSX parsing in .js files
//       }
//     }
//   }
// });
