const AuditLog = require('../models/AuditLog');

module.exports = function auditLogger() {
  return async function(req, res, next) {
    const start = Date.now();
    res.on('finish', async () => {
      try {
        const action = `${req.method} ${req.path}`;
        const userId = req.user?.userId;
        await AuditLog.create({
          userId,
          action,
          ip: req.ip,
          userAgent: req.get('User-Agent'),
          metadata: { status: res.statusCode, durationMs: Date.now() - start }
        });
      } catch {}
    });
    next();
  };
}


