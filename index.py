import discord
import json
from discord import app_commands
from huggingface_hub import HfApi, HfFolder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from googletrans import Translator

f = open('./config.json')
data = json.load(f)

MY_GUILD = discord.Object(id=data['discord_guild_id'])

class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        # This copies the global commands over to our guild.
        self.tree.copy_global_to(guild=MY_GUILD)
        await self.tree.sync(guild=MY_GUILD)

intents = discord.Intents.default()
client = MyClient(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user} (ID: {client.user.id})')

@client.tree.command()
@app_commands.describe(
    message='The message you want to translate',
)
async def translate(interaction: discord.Interaction, message: str) -> None:
    """Ask chatbot to translate message to NLLB and google translate"""
    await interaction.response.defer()

    try:
        
        embedVar = nllb_gtrans_translate(message)
        await interaction.followup.send(embed=embedVar)

    except Exception as e:
        print(e)
        await interaction.followup.send("Failed to translate!")

def nllb_gtrans_translate(message: str):

    inputs = tokenizer(message, return_tensors="pt", padding = True)

    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["khm_Khmer"], max_length=100)
    res_nllb = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    translator = Translator()
    translation = translator.translate(message, dest='km')
    res_gtrans = translation.text

    embedVar = discord.Embed(title="Result", color=0x00ff00)
    embedVar.add_field(name="input", value=message, inline=False)
    embedVar.add_field(name="nllb", value=res_nllb, inline=False)
    embedVar.add_field(name="gtrans", value=res_gtrans, inline=False)

    return embedVar

def login_hugging_face(token: str) -> None:
    """
    Loging to Hugging Face portal with a given token.
    """
    api = HfApi()
    api.set_access_token(token)
    folder = HfFolder()
    folder.save_token(token)

    return None

# Check if value set and login
if ('hugging_face_token' in data and len(data['hugging_face_token']) > 0):
    login_hugging_face(data['hugging_face_token'])

# Load tokenizer and model
print('Loading tokenizer and model')
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True)
print('Finished loading!')

client.run(data['discord_bot_token'])
