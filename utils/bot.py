import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get token from environment
TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable not set!")
    raise ValueError("TELEGRAM_BOT_TOKEN is required")

# Store user states
user_states = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler."""
    keyboard = [
        [
            InlineKeyboardButton("🖼️ Convert Image", callback_data='convert'),
            InlineKeyboardButton("🎨 Generate Image", callback_data='generate')
        ],
        [
            InlineKeyboardButton("🔗 Shorten URL", callback_data='shorten'),
            InlineKeyboardButton("❓ Help", callback_data='help')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🎨 *Welcome to DailyDroppBot!*\n\n"
        "Your all-in-one image and link management bot.\n\n"
        "✨ *Features:*\n"
        "🖼️ Image Conversion\n"
        "🎨 AI Image Generation\n"
        "🔗 URL Shortening\n\n"
        "Click a button below to get started:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    data = query.data
    
    if data == 'convert':
        user_states[user_id] = 'waiting_for_image'
        await query.edit_message_text(
            "🖼️ *Image Conversion*\n\n"
            "Please send me an image (photo) you want to convert.\n"
            "Supported formats: PNG, JPG, WEBP, GIF, BMP, TIFF"
        )
    elif data == 'generate':
        user_states[user_id] = 'waiting_for_prompt'
        await query.edit_message_text(
            "🎨 *AI Image Generation*\n\n"
            "Please describe the image you want to create.\n\n"
            "Example: 'A beautiful sunset over mountains with purple sky'"
        )
    elif data == 'shorten':
        user_states[user_id] = 'waiting_for_url'
        await query.edit_message_text(
            "🔗 *URL Shortening*\n\n"
            "Please send me the URL you want to shorten.\n"
            "Example: https://example.com/very/long/url"
        )
    elif data == 'help':
        await query.edit_message_text(
            "📖 *Help*\n\n"
            "• /start - Start the bot\n"
            "• /help - Show help\n\n"
            "Use the buttons to access features."
        )

async def handle_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages."""
    user_id = update.effective_user.id
    state = user_states.get(user_id)
    
    if not state:
        await update.message.reply_text("Please use /start to begin!")
        return
    
    if state == 'waiting_for_image':
        if update.message.photo:
            await update.message.reply_text("✅ Image received! Converting...")
            # Add your conversion logic here
        else:
            await update.message.reply_text("Please send a photo!")
    
    elif state == 'waiting_for_prompt':
        prompt = update.message.text
        if len(prompt) >= 5:
            await update.message.reply_text(f"🎨 Generating image from: {prompt[:50]}...")
            # Add your generation logic here
        else:
            await update.message.reply_text("Please provide a longer prompt (5+ characters)!")
    
    elif state == 'waiting_for_url':
        url = update.message.text
        if url.startswith('http://') or url.startswith('https://'):
            await update.message.reply_text(f"🔗 Shortening URL...")
            # Add your URL shortening logic here
        else:
            await update.message.reply_text("Please send a valid URL (starts with http:// or https://)")

def main():
    """Start the bot."""
    try:
        # Create application
        application = Application.builder().token(TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", start))
        application.add_handler(CallbackQueryHandler(handle_button))
        application.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, handle_messages))
        
        # Start polling
        logger.info("🤖 DailyDroppBot is starting...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise

if __name__ == '__main__':
    main()
