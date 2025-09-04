const logger = require('../utils/logger');

function parseList(value) {
  if (!value) return [];
  return value.split(',').map(v => v.trim()).filter(Boolean);
}

module.exports = function ipFilter() {
  const allow = parseList(process.env.IP_ALLOW_LIST);
  const deny = parseList(process.env.IP_DENY_LIST);

  return function(req, res, next) {
    const ip = (req.headers['x-forwarded-for'] || req.connection.remoteAddress || '').toString();

    if (deny.length && deny.some(block => ip.includes(block))) {
      logger.warn(`Blocked IP ${ip}`);
      return res.status(403).json({ success: false, message: 'Forbidden' });
    }
    if (allow.length && !allow.some(ok => ip.includes(ok))) {
      logger.warn(`Denied IP ${ip} not in allow list`);
      return res.status(403).json({ success: false, message: 'Forbidden' });
    }
    next();
  };
};


