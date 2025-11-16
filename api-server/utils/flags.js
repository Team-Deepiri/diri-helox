const flags = {
  enableAgentStreaming: process.env.FLAG_AGENT_STREAMING === 'true',
  usePythonAgent: process.env.FLAG_USE_CYREX === 'true',
  enableRecommendations: process.env.FLAG_RECS === 'true'
};

module.exports = {
  getAll: () => flags,
  isOn: (key) => !!flags[key]
};


