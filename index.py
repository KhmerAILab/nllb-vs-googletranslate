import discord
import json
from discord.ext import commands
from huggingface_hub import HfApi, HfFolder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from googletrans import Translator

f = open('./config.json')
data = json.load(f)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def translate(ctx, arg):
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True)
    article = arg
    inputs = tokenizer(article, return_tensors="pt")

    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["khm_Khmr"], max_length=100)
    res_nllb = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    translator = Translator()
    translation = translator.translate("test", dest='km')
    res_gtrans = translation.text

    embedVar = discord.Embed(title="Result", color=0x00ff00)
    embedVar.add_field(name="nllb", value=res_nllb, inline=False)
    embedVar.add_field(name="gtrans", value=res_gtrans, inline=False)


    await ctx.send(embed=embedVar)

def login_hugging_face(token: str) -> None:
    """
    Loging to Hugging Face portal with a given token.
    """
    api = HfApi()
    api.set_access_token(token)
    folder = HfFolder()
    folder.save_token(token)

    return None
login_hugging_face(data['hugging_face_token'])


bot.run(data['discord_bot_token'])
