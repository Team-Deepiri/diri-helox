// Basic NoSQL injection guard and payload sanitizer
module.exports = function sanitize() {
  return function(req, res, next) {
    const strip = (obj) => {
      if (!obj || typeof obj !== 'object') return obj;
      if (Array.isArray(obj)) return obj.map(strip);
      const clean = {};
      for (const [k, v] of Object.entries(obj)) {
        if (k.startsWith('$') || k.includes('.')) continue;
        clean[k] = strip(v);
      }
      return clean;
    };
    if (req.body) req.body = strip(req.body);
    if (req.query) req.query = strip(req.query);
    if (req.params) req.params = strip(req.params);
    next();
  };
}


