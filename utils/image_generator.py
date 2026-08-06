import os
import io
import base64
import logging
import requests
import aiohttp
import asyncio
from typing import Optional, Dict, Any, Union
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ImageGenerator:
    """
    Advanced text-to-image generator with multiple AI providers.
    Supports: Hugging Face, OpenAI DALL-E, Stability AI, Replicate, and mock generation.
    """
    
    def __init__(self, config: Dict[str, str]):
        """
        Initialize the image generator with API keys.
        
        Args:
            config: Dictionary containing API keys and settings
        """
        self.config = config
        self.providers = {
            'huggingface': self._generate_huggingface,
            'openai': self._generate_openai,
            'stability': self._generate_stability_ai,
            'replicate': self._generate_replicate,
            'mock': self._generate_mock
        }
        self.active_providers = self._get_active_providers()
        
    def _get_active_providers(self) -> list:
        """Get list of providers with available API keys."""
        available = []
        
        if self.config.get('HUGGINGFACE_API_KEY'):
            available.append('huggingface')
        if self.config.get('OPENAI_API_KEY'):
            available.append('openai')
        if self.config.get('STABILITY_API_KEY'):
            available.append('stability')
        if self.config.get('REPLICATE_API_TOKEN'):
            available.append('replicate')
        
        # Always include mock as fallback
        available.append('mock')
        
        return available
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        provider: Optional[str] = None,
        **kwargs
    ) -> Optional[bytes]:
        """
        Generate an image from text prompt.
        
        Args:
            prompt: Text description of the image
            negative_prompt: What to avoid in the image
            width: Image width
            height: Image height
            num_images: Number of images to generate
            provider: Specific provider to use (optional)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Generated image as bytes, or None if failed
        """
        try:
            # Validate prompt
            if not prompt or len(prompt) < 5:
                logger.error("Prompt too short")
                return None
            
            # Choose provider
            if provider and provider in self.providers:
                providers = [provider]
            else:
                providers = self.active_providers
            
            # Try each provider until one succeeds
            for prov in providers:
                try:
                    logger.info(f"Generating image with {prov}")
                    result = await self.providers[prov](
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_images=num_images,
                        **kwargs
                    )
                    
                    if result:
                        logger.info(f"Successfully generated image with {prov}")
                        return result
                        
                except Exception as e:
                    logger.error(f"Provider {prov} failed: {e}")
                    continue
            
            # If all providers fail, use mock
            return await self._generate_mock(prompt)
            
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return None
    
    # ============ PROVIDER IMPLEMENTATIONS ============
    
    async def _generate_huggingface(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        **kwargs
    ) -> Optional[bytes]:
        """Generate using Hugging Face Inference API."""
        try:
            api_key = self.config.get('HUGGINGFACE_API_KEY')
            if not api_key:
                raise ValueError("Hugging Face API key not configured")
            
            # Choose model
            model = kwargs.get('model', 'stabilityai/stable-diffusion-2-1')
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": negative_prompt or "",
                    "width": width,
                    "height": height,
                    "num_inference_steps": kwargs.get('steps', 30),
                    "guidance_scale": kwargs.get('guidance_scale', 7.5)
                }
            }
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.read()
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"Hugging Face API error: {response.status} - {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error("Hugging Face API timeout")
            return None
        except Exception as e:
            logger.error(f"Hugging Face generation error: {e}")
            return None
    
    async def _generate_openai(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        **kwargs
    ) -> Optional[bytes]:
        """Generate using OpenAI DALL-E API."""
        try:
            api_key = self.config.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not configured")
            
            import openai
            openai.api_key = api_key
            
            # DALL-E 2 or 3
            model = kwargs.get('model', 'dall-e-2')
            
            # Adjust size based on model
            if model == 'dall-e-3':
                size = "1024x1024"
                quality = kwargs.get('quality', 'standard')
            else:
                # DALL-E 2 supports 256x256, 512x512, 1024x1024
                if width <= 256 and height <= 256:
                    size = "256x256"
                elif width <= 512 and height <= 512:
                    size = "512x512"
                else:
                    size = "1024x1024"
                quality = "standard"
            
            # Generate image
            response = openai.Image.create(
                prompt=prompt,
                n=min(num_images, 4),  # DALL-E max 4 images
                size=size,
                quality=quality,
                response_format="url"
            )
            
            # Download first image
            if response and response['data']:
                image_url = response['data'][0]['url']
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as img_response:
                        if img_response.status == 200:
                            return await img_response.read()
            
            return None
            
        except ImportError:
            logger.error("OpenAI package not installed")
            return None
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return None
    
    async def _generate_stability_ai(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        **kwargs
    ) -> Optional[bytes]:
        """Generate using Stability AI API."""
        try:
            api_key = self.config.get('STABILITY_API_KEY')
            if not api_key:
                raise ValueError("Stability AI API key not configured")
            
            # API endpoint
            api_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare payload
            payload = {
                "text_prompts": [
                    {"text": prompt, "weight": 1.0}
                ],
                "cfg_scale": kwargs.get('cfg_scale', 7),
                "height": height,
                "width": width,
                "samples": min(num_images, 4),
                "steps": kwargs.get('steps', 30)
            }
            
            if negative_prompt:
                payload["text_prompts"].append({
                    "text": negative_prompt,
                    "weight": -1.0
                })
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and data.get('artifacts'):
                            # Decode base64 image
                            image_data = base64.b64decode(data['artifacts'][0]['base64'])
                            return image_data
                    else:
                        error_text = await response.text()
                        logger.error(f"Stability AI API error: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Stability AI generation error: {e}")
            return None
    
    async def _generate_replicate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        **kwargs
    ) -> Optional[bytes]:
        """Generate using Replicate API."""
        try:
            api_token = self.config.get('REPLICATE_API_TOKEN')
            if not api_token:
                raise ValueError("Replicate API token not configured")
            
            # Choose model
            model = kwargs.get('model', 'stability-ai/sdxl')
            version = kwargs.get('version', '39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b')
            
            # Prepare input
            input_data = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "width": width,
                "height": height,
                "num_outputs": min(num_images, 4),
                "guidance_scale": kwargs.get('guidance_scale', 7.5),
                "num_inference_steps": kwargs.get('steps', 30)
            }
            
            # Create prediction
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Token {api_token}",
                    "Content-Type": "application/json"
                }
                
                # Start prediction
                async with session.post(
                    "https://api.replicate.com/v1/predictions",
                    headers=headers,
                    json={
                        "version": version,
                        "input": input_data
                    }
                ) as response:
                    if response.status != 201:
                        error_text = await response.text()
                        logger.error(f"Replicate API error: {response.status} - {error_text}")
                        return None
                    
                    prediction = await response.json()
                    prediction_url = prediction['urls']['get']
                
                # Poll for result
                for _ in range(30):  # Max 30 attempts (about 30 seconds)
                    await asyncio.sleep(1)
                    
                    async with session.get(
                        prediction_url,
                        headers=headers
                    ) as status_response:
                        if status_response.status != 200:
                            continue
                        
                        status_data = await status_response.json()
                        
                        if status_data['status'] == 'succeeded':
                            # Download image
                            image_url = status_data['output'][0]
                            async with session.get(image_url) as img_response:
                                if img_response.status == 200:
                                    return await img_response.read()
                            break
                        elif status_data['status'] == 'failed':
                            logger.error(f"Replicate prediction failed: {status_data}")
                            break
                
                return None
                
        except Exception as e:
            logger.error(f"Replicate generation error: {e}")
            return None
    
    async def _generate_mock(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        **kwargs
    ) -> Optional[bytes]:
        """
        Generate a mock image for testing.
        Creates a gradient image with the prompt text.
        """
        try:
            # Create gradient image
            image = Image.new('RGB', (width, height))
            pixels = image.load()
            
            # Generate random gradient
            import random
            colors = [
                (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)),
                (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            ]
            
            for x in range(width):
                for y in range(height):
                    ratio = x / width
                    r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                    g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                    b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                    pixels[x, y] = (r, g, b)
            
            # Add border
            draw = ImageDraw.Draw(image)
            draw.rectangle([(0, 0), (width-1, height-1)], outline='white', width=5)
            
            # Add text
            try:
                font_size = min(width, height) // 20
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Wrap prompt text
            words = prompt.split()
            lines = []
            current_line = ""
            max_chars = width // (font_size // 2)
            
            for word in words:
                if len(current_line + " " + word) <= max_chars:
                    current_line += " " + word if current_line else word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # Draw text
            y_offset = height // 2 - len(lines) * font_size // 2
            for line in lines:
                draw.text(
                    (width//2, y_offset),
                    line,
                    fill='white',
                    font=font,
                    anchor='mm',
                    stroke_width=2,
                    stroke_fill='black'
                )
                y_offset += font_size + 5
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            draw.text(
                (width-10, height-20),
                f"DailyDroppBot {timestamp}",
                fill='white',
                font=font,
                anchor='rs',
                stroke_width=1,
                stroke_fill='black'
            )
            
            # Save to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='PNG', quality=95)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Mock generation error: {e}")
            return None

# ============ CONVENIENCE FUNCTIONS ============

class ImageGeneratorFactory:
    """Factory for creating image generator instances."""
    
    _instance = None
    
    @classmethod
    def get_generator(cls, config: Dict[str, str]) -> ImageGenerator:
        """Get or create an image generator instance."""
        if cls._instance is None:
            cls._instance = ImageGenerator(config)
        return cls._instance

# ============ ASYNC WRAPPER FOR SYNC USE ============

async def generate_image_async(
    prompt: str,
    config: Dict[str, str],
    negative_prompt: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    **kwargs
) -> Optional[bytes]:
    """
    Async wrapper for image generation.
    
    Args:
        prompt: Text description
        config: Configuration dictionary with API keys
        negative_prompt: What to avoid
        width: Image width
        height: Image height
        **kwargs: Additional parameters
    
    Returns:
        Generated image as bytes
    """
    generator = ImageGeneratorFactory.get_generator(config)
    return await generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        **kwargs
    )

# ============ TELEGRAM BOT INTEGRATION ============

class TelegramImageGenerator:
    """
    Image generator specifically designed for Telegram bot integration.
    Handles async operations and progress updates.
    """
    
    def __init__(self, bot, config: Dict[str, str]):
        """
        Initialize with Telegram bot instance.
        
        Args:
            bot: Telegram bot instance
            config: Configuration with API keys
        """
        self.bot = bot
        self.generator = ImageGeneratorFactory.get_generator(config)
        self.config = config
    
    async def generate_and_send(
        self,
        chat_id: Union[int, str],
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        **kwargs
    ) -> bool:
        """
        Generate image and send directly to Telegram chat.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Send progress message
            progress_msg = await self.bot.send_message(
                chat_id=chat_id,
                text=f"🎨 Generating image...\n\n📝 Prompt: {prompt[:100]}...\n⏳ Please wait..."
            )
            
            # Generate image
            image_data = await self.generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_images=num_images,
                **kwargs
            )
            
            if image_data:
                # Send generated image
                filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                await self.bot.send_document(
                    chat_id=chat_id,
                    document=io.BytesIO(image_data),
                    filename=filename,
                    caption=f"🎨 *Generated Image*\n\n📝 {prompt}\n\n"
                           f"📐 {width}x{height} • Generated by DailyDroppBot",
                    parse_mode='Markdown'
                )
                
                # Delete progress message
                await progress_msg.delete()
                return True
            else:
                await progress_msg.edit_text(
                    "❌ Failed to generate image. Please try again later."
                )
                return False
                
        except Exception as e:
            logger.error(f"Telegram generation error: {e}")
            await self.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Generation failed: {str(e)}"
            )
            return False
    
    async def generate_with_options(
        self,
        chat_id: Union[int, str],
        prompt: str
    ) -> None:
        """
        Generate image with interactive options (for inline mode).
        """
        # Send options keyboard
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        keyboard = [
            [
                InlineKeyboardButton("📏 512x512", callback_data="gen_512"),
                InlineKeyboardButton("📏 768x768", callback_data="gen_768"),
                InlineKeyboardButton("📏 1024x1024", callback_data="gen_1024")
            ],
            [
                InlineKeyboardButton("🎨 Standard", callback_data="gen_standard"),
                InlineKeyboardButton("🎨 Detailed", callback_data="gen_detailed"),
                InlineKeyboardButton("🎨 Artistic", callback_data="gen_artistic")
            ],
            [
                InlineKeyboardButton("✅ Generate", callback_data=f"gen_execute_{prompt[:50]}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await self.bot.send_message(
            chat_id=chat_id,
            text=f"🎨 *Generate Image*\n\n"
                 f"📝 Prompt: {prompt}\n\n"
                 "Choose options:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
