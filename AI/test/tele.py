from telegram.ext import Updater, CommandHandler

# Token lấy từ BotFather
TOKEN = "7850063944:AAHoZeCVGu2PuRswtzKWqhwm3WuuGlzlbEg"

def start(update, context):
    chat_id = update.message.chat_id
    update.message.reply_text(f"Chat ID của bạn là: {chat_id}")

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    # Lệnh /start
    dp.add_handler(CommandHandler("start", start))

    # Chạy bot
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()