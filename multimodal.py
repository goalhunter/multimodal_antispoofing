import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, BertModel, BertTokenizer, WhisperForConditionalGeneration, WhisperProcessor

class CrossModalFusion(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=768, num_heads=8):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.common_dim = 512

        # Projection layers
        self.audio_proj = nn.Linear(audio_dim, self.common_dim)
        self.text_proj = nn.Linear(text_dim, self.common_dim)

        # Single attention layer with common dimension
        self.attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Final fusion layer (common_dim * 2 because of concatenation)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.common_dim, self.common_dim)
        )

    def forward(self, audio_features, text_features):
        # Project to common dimension
        audio_proj = self.audio_proj(audio_features)  # [batch, common_dim]
        text_proj = self.text_proj(text_features)     # [batch, common_dim]
        
        # Add sequence dimension for attention
        audio_proj = audio_proj.unsqueeze(1)  # [batch, 1, common_dim]
        text_proj = text_proj.unsqueeze(1)    # [batch, 1, common_dim]

        # Cross attention
        attended_features, _ = self.attention(
            query=audio_proj,
            key=text_proj,
            value=text_proj
        )  # [batch, 1, common_dim]
        
        # Concatenate along feature dimension
        fused = torch.cat([audio_proj, attended_features], dim=-1)  # [batch, 1, common_dim*2]
        fused = fused.squeeze(1)  # [batch, common_dim*2]
        
        # Final fusion
        output = self.fusion_layer(fused)  # [batch, common_dim]
        
        return output

class MultiModalDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        
        # Define dimensions
        audio_dim = 1024
        text_dim = 768
        common_dim = 512

        # Add your fusion layers and classifier
        self.fusion = CrossModalFusion(
            audio_dim=audio_dim,
            text_dim=text_dim,
            num_heads=8
        )

        self.classifier = nn.Sequential(
            nn.Linear(common_dim, common_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(common_dim//2, 2)  # Binary classification
        )
        # MLP classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(1024, 512),  # wav2vec output dimension -> hidden dim
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, 2)      # Binary classification
        # )
        
    def forward(self, audio):
        # Get audio features from wav2vec (this should work fine)
        audio_features = self.audio_encoder(audio).last_hidden_state.mean(dim=1)  # [batch, time, dim] -> [batch, dim]
        
        # Process audio for Whisper
        input_features = self.whisper_processor(
            audio.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).to(audio.device)

        # Generate text tokens
        generated_ids = self.asr_model.generate(
            input_features.input_features,
            max_length=225,
            language='en'
        )
        
        # Convert tokens to BERT input
        text = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)
        text_inputs = self.bert_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(audio.device)

        # Get text features
        text_features = self.text_encoder(**text_inputs).last_hidden_state.mean(dim=1)  # [batch, seq_len, dim] -> [batch, dim]

        # Fusion and classification
        fused = self.fusion(audio_features, text_features)
        output = self.classifier(fused)
        
        return output