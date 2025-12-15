module.exports = {
  apps : [{
    name: 'generation',
    script: 'serve.py',
    interpreter: '/venv/404-base-miner/bin/python',
    args: '--port 8094'
  }]
};
