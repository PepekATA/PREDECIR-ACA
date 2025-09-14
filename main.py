The user requires a Python script for an AI Crypto Trading Bot named "PAPA-DINERO". The bot's core strategy is "Never Sell at Loss - Always Profit". It should be developed using API integration, targeting an AI/ML Specialist, and include API integration as a key feature. The script should be in English and utilize a specific set of Python libraries.

```python
#!/usr/bin/env python3
"""
PAPA-DINERO - AI Crypto Trading Bot
Never Sell at Loss - Always Profit Strategy

Main entry point for the trading bot.
Handles initialization, coordination, and 24/7 operation.
"""

import os
import sys
import asyncio
import signal
import logging
import json
from datetime import datetime
import traceback
from pathlib import Path

# Configure module path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Basic logging configuration
log_dir = current_dir / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'papa_dinero.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("PAPA-DINERO")

class PapaDineroBot:
    """Main trading bot that coordinates all modules"""
    
    def __init__(self):
        self.running = False
        self.cycle_count = 0
        
        # Component states
        self.components = {}
        self.api_connected = False
        self.credentials = None
        
        # Create necessary directories
        self.create_directories()
        
        logger.info("🤖 PAPA-DINERO Bot initialized")
        logger.info("💎 Strategy: NEVER SELL AT LOSS - ALWAYS PROFIT")
    
    def create_directories(self):
        """Create necessary directories"""
        directories = ['data', 'logs', 'config']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_credentials_from_env(self):
        """Get credentials from environment variables"""
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        paper_trading = os.getenv('PAPER_TRADING', 'True').lower() == 'true'
        
        if not api_key or not api_secret:
            logger.error("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY variables not found")
            return None
        
        self.credentials = {
            'api_key': api_key,
            'api_secret': api_secret,
            'paper_trading': paper_trading,
            'base_url': 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
        }
        
        logger.info(f"✅ Credentials loaded - Mode: {'Paper' if paper_trading else 'Live'}")
        return self.credentials
    
    async def initialize_components(self):
        """Initialize bot components securely"""
        try:
            # Verify credentials
            if not self.get_credentials_from_env():
                raise Exception("Credentials not available")
            
            # Import modules dynamically to handle errors
            try:
                from modules.memory_system import MemorySystem
                self.components['memory'] = MemorySystem()
                logger.info("✅ Memory system initialized")
            except ImportError as e:
                logger.warning(f"⚠️ Memory system not available: {e}")
                self.components['memory'] = None
            
            try:
                from modules.trading_engine import TradingEngine
                if self.components['memory']:
                    self.components['trading'] = TradingEngine(self.components['memory'])
                    
                    # Connect API
                    api_connected = self.components['trading'].initialize_api(
                        self.credentials['api_key'],
                        self.credentials['api_secret'],
                        self.credentials['paper_trading']
                    )
                    
                    if api_connected:
                        self.api_connected = True
                        logger.info("✅ Trading engine connected to Alpaca")
                    else:
                        raise Exception("Error connecting to Alpaca API")
                else:
                    raise Exception("Memory system required for trading engine")
                    
            except Exception as e:
                logger.error(f"❌ Trading engine not available: {e}")
                self.components['trading'] = None
            
            # Other optional components
            try:
                from modules.ai_predictor import AIPredictor
                if self.components['memory']:
                    self.components['ai'] = AIPredictor(self.components['memory'])
                    logger.info("✅ AI Predictor initialized")
            except ImportError as e:
                logger.warning(f"⚠️ AI Predictor not available: {e}")
                self.components['ai'] = None
            
            try:
                from modules.portfolio_manager import PortfolioManager
                self.components['portfolio'] = PortfolioManager()
                logger.info("✅ Portfolio Manager initialized")
            except ImportError as e:
                logger.warning(f"⚠️ Portfolio Manager not available: {e}")
                self.components['portfolio'] = None
            
            # Create a state file to indicate the bot is active
            await self.create_bot_state_file()
            
            return self.api_connected
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            return False
    
    async def create_bot_state_file(self):
        """Create state file for the dashboard"""
        try:
            state = {
                'status': 'active' if self.api_connected else 'demo',
                'initialized_at': datetime.now().isoformat(),
                'components': {
                    'trading_engine': self.components.get('trading') is not None,
                    'ai_predictor': self.components.get('ai') is not None,
                    'memory_system': self.components.get('memory') is not None,
                    'portfolio_manager': self.components.get('portfolio') is not None,
                    'api_connected': self.api_connected
                },
                'credentials_found': self.credentials is not None,
                'paper_trading': self.credentials.get('paper_trading', True) if self.credentials else True
            }
            
            with open('data/bot_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info("✅ State file created")
            
        except Exception as e:
            logger.error(f"❌ Error creating state file: {e}")
    
    def get_active_symbols(self):
        """Active symbols for trading"""
        return [
            'BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'ADAUSD',
            'DOTUSD', 'MATICUSD', 'LINKUSD', 'UNIUSD', 'AAVEUSD'
        ]
    
    async def run_trading_cycle(self):
        """Execute a complete trading cycle"""
        if not self.api_connected or not self.components.get('trading'):
            logger.warning("⚠️ API not connected, skipping trading cycle")
            return {'status': 'skipped', 'reason': 'api_not_connected'}
        
        try:
            self.cycle_count += 1
            start_time = datetime.now()
            
            logger.info(f"🔄 Cycle #{self.cycle_count} - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get active symbols
            symbols = self.get_active_symbols()
            
            # Execute automatic trading cycle
            trading_engine = self.components['trading']
            ai_predictor = self.components.get('ai')
            
            if ai_predictor:
                # Trading with AI
                results = trading_engine.auto_trading_cycle(symbols, ai_predictor, max_positions=5)
            else:
                # Basic trading without AI
                logger.info("🤖 Trading without AI - using basic technical analysis")
                results = await self.basic_trading_cycle(symbols)
            
            # Update state file
            await self.update_bot_state(results)
            
            cycle_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"⏱️ Cycle completed in {cycle_time:.2f}s - Actions taken: {len(results.get('actions_taken', []))}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}
    
    async def basic_trading_cycle(self, symbols):
        """Basic trading cycle without AI"""
        actions = []
        
        try:
            trading_engine = self.components['trading']
            
            # Review existing positions
            for symbol in list(trading_engine.positions.keys()):
                if trading_engine.positions[symbol]['status'] == 'active':
                    try:
                        # Get current price (simulated for demo)
                        current_price = 45000  # Demo price
                        
                        analysis = trading_engine.analyze_position_profitability(symbol, current_price)
                        
                        if analysis['can_sell'] and analysis['recommended_action'] == 'sell_profit':
                            # Attempt to sell (only with profit)
                            logger.info(f"💰 Attempting to sell with profit: {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
            
            return {
                'timestamp': datetime.now().isoformat(),
                'actions_taken': actions,
                'positions_analyzed': len(trading_engine.positions),
                'mode': 'basic_trading'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in basic trading: {e}")
            return {'error': str(e)}
    
    async def update_bot_state(self, trading_results):
        """Update bot state"""
        try:
            state = {
                'status': 'active' if self.api_connected else 'demo',
                'last_update': datetime.now().isoformat(),
                'total_cycles': self.cycle_count,
                'api_connected': self.api_connected,
                'last_trading_results': trading_results,
                'components': {
                    'trading_engine': self.components.get('trading') is not None,
                    'ai_predictor': self.components.get('ai') is not None,
                    'memory_system': self.components.get('memory') is not None,
                    'portfolio_manager': self.components.get('portfolio') is not None
                }
            }
            
            with open('data/bot_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error updating state: {e}")
    
    async def run_main_loop(self):
        """Main bot loop"""
        logger.info("🚀 Starting main loop...")
        self.running = True
        
        while self.running:
            try:
                # Execute trading cycle
                await self.run_trading_cycle()
                
                # Wait 60 seconds between cycles
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Manual interruption")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(120)  # Wait longer if there's an error
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down bot...")
        self.running = False
        
        try:
            # Save final state
            if self.components.get('trading'):
                self.components['trading'].save_trading_state()
            
            # Update state to inactive
            with open('data/bot_state.json', 'w') as f:
                json.dump({
                    'status': 'stopped',
                    'stopped_at': datetime.now().isoformat(),
                    'total_cycles': self.cycle_count
                }, f, indent=2)
            
            logger.info("✅ Bot shut down successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

async def main():
    """Main function"""
    bot = PapaDineroBot()
    
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"🛑 Signal {signum} received")
        asyncio.create_task(bot.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize components
        if await bot.initialize_components():
            logger.info("✅ Bot initialized, starting trading...")
            await bot.run_main_loop()
        else:
            logger.warning("⚠️ Bot started in limited mode (no API)")
            # Create state file for demo
            await bot.create_bot_state_file()
            # Keep alive for Streamlit
            while True:
                await asyncio.sleep(60)
                await bot.create_bot_state_file()  # Update state
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Manual exit")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        traceback.print_exc()
```
