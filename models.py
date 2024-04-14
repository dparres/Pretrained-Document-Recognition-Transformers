import torch
from torch import nn
from torch.nn import functional as F
from transformers import VisionEncoderDecoderModel, DonutProcessor, VisionEncoderDecoderConfig

DONUT_WEIGHTS = "naver-clova-ix/donut-base"

######################################################
# Swin + CTC
######################################################

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Swin_CTC(nn.Module):

    def __init__(self, vocab_size=100):
        super().__init__()

        # Swin Config
        HEIGHT = 2560
        WIDTH = 1920
        config = VisionEncoderDecoderConfig.from_pretrained(DONUT_WEIGHTS)
        config.encoder.image_size = [HEIGHT, WIDTH]
        
        # Image Processor
        self.processor = DonutProcessor.from_pretrained(DONUT_WEIGHTS)
        self.processor.image_processor.size = [WIDTH, HEIGHT]
        self.processor.image_processor.do_align_long_axis = False

        # Swin Encoder
        self.swin_encoder = VisionEncoderDecoderModel.from_pretrained(DONUT_WEIGHTS, config=config).encoder
        self.swin_encoder.pooler = Identity()

        # Fully-connected Layer to Vocab
        self.projection_V = nn.Linear(1024, vocab_size+1) # classes + blank token

    def forward(self, x, targets=None, target_lengths=None):

        x = self.swin_encoder(x).last_hidden_state # (b, 4800, 1024)
        x = self.projection_V(x) # (b, 4800,1024) to (b, 4800, V)

        if targets is not None:
            x = x.permute(1, 0, 2)
            loss = self.ctc_loss(x,targets, target_lengths)
            return x, loss

        return x, None

    @staticmethod
    def ctc_loss(x, targets, target_lengths):
        batch_size = x.size(1)
       
        log_probs = F.log_softmax(x, 2)
    
        input_lengths = torch.full(
            size=(batch_size,), 
            fill_value=log_probs.size(0), 
            dtype=torch.int32
        )

        loss = nn.CTCLoss(blank=0)(
            log_probs, targets, input_lengths, target_lengths
        )

        return loss


######################################################
# Vision Encoder-Decoder (VED)
######################################################

class VED(nn.Module):

    def __init__(self):
        super().__init__()

        # VED Config
        HEIGHT = 2560
        WIDTH = 1920
        MAX_LENGTH = 768
        config = VisionEncoderDecoderConfig.from_pretrained(DONUT_WEIGHTS)
        config.encoder.image_size = [HEIGHT, WIDTH]
        config.decoder.max_length = MAX_LENGTH
        
        # Image Processor
        self.processor = DonutProcessor.from_pretrained(DONUT_WEIGHTS)
        self.processor.image_processor.size = [WIDTH, HEIGHT]
        self.processor.image_processor.do_align_long_axis = False

        # VED Model
        self.model = VisionEncoderDecoderModel.from_pretrained(DONUT_WEIGHTS, config=config)

        # Params for Transformer Decoder
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        # set <s_synthdog> token=57524
        self.model.config.decoder_start_token_id = 57524

    def forward(self, x, labels):

        outputs = self.model(x, labels=labels)    
        return outputs.loss