import re
import torch
from torch import nn
from torch.nn import functional as F
from transformers import VisionEncoderDecoderModel, DonutProcessor, VisionEncoderDecoderConfig

import paths

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
        HEIGHT = paths.HEIGHT
        WIDTH = paths.WIDTH
        config = VisionEncoderDecoderConfig.from_pretrained(paths.DONUT_WEIGHTS)
        config.encoder.image_size = [HEIGHT, WIDTH]
        
        # Image Processor
        self.processor = DonutProcessor.from_pretrained(paths.DONUT_WEIGHTS)
        self.processor.image_processor.size = [WIDTH, HEIGHT]
        self.processor.image_processor.do_align_long_axis = False

        # Swin Encoder
        self.swin_encoder = VisionEncoderDecoderModel.from_pretrained(paths.DONUT_WEIGHTS, config=config).encoder
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
    
    def inference_one_sample(self, x, seq_to_text):

        x, _ = self(x) # forward of Swin+CTC model

        x = x.permute(1, 0, 2)

        x, xs = x, [x.size(0)] * x.size(1)
        x = x.detach()
        
        x = torch.nn.functional.log_softmax(x, 2)
        
        # Transform to list of size = batch_size
        x = [x[: xs[i], i, :] for i in range(len(xs))]
        x = [x_n.max(dim=1) for x_n in x]

        # Get symbols and probabilities
        probs = [x_n.values.exp() for x_n in x]
        x = [x_n.indices for x_n in x]

        # Remove consecutive symbols
        # Keep track of counts of consecutive symbols. Example: [0, 0, 0, 1, 2, 2] => [3, 1, 2]
        counts = [torch.unique_consecutive(x_n, return_counts=True)[1] for x_n in x]

        # Select indexes to keep. Example: [0, 3, 4] (always keep the first index, then use cumulative sum of counts tensor)
        zero_tensor = torch.tensor([0], device=x.device)
        idxs = [torch.cat((zero_tensor, count.cumsum(0)[:-1])) for count in counts]

        # Keep only non consecutive symbols and their associated probabilities
        x = [x[i][idxs[i]] for i in range(len(x))]
        probs = [probs[i][idxs[i]] for i in range(len(x))]

        # Remove blank symbols
        # Get index for non blank symbols
        idxs = [torch.nonzero(x_n, as_tuple=True) for x_n in x]

        # Keep only non blank symbols and their associated probabilities
        x = [x[i][idxs[i]] for i in range(len(x))]
        probs = [probs[i][idxs[i]] for i in range(len(x))]

        # Save results
        out = {}
        out["hyp"] = [x_n.tolist() for x_n in x]

        # Return char-based probability
        out["prob-htr-char"] = [prob.tolist() for prob in probs]

        text = ""
        for i in out["hyp"][0]:
            text += seq_to_text[i]

        return text


######################################################
# Vision Encoder-Decoder (VED)
######################################################

class VED(nn.Module):

    def __init__(self):
        super().__init__()

        # VED Config
        HEIGHT = paths.HEIGHT
        WIDTH = paths.WIDTH
        self.MAX_LENGTH = paths.MAX_LENGTH
        config = VisionEncoderDecoderConfig.from_pretrained(paths.DONUT_WEIGHTS)
        config.encoder.image_size = [HEIGHT, WIDTH]
        config.decoder.max_length = self.MAX_LENGTH
        
        # Image Processor
        self.processor = DonutProcessor.from_pretrained(paths.DONUT_WEIGHTS)
        self.processor.image_processor.size = [WIDTH, HEIGHT]
        self.processor.image_processor.do_align_long_axis = False

        # VED Model
        self.model = VisionEncoderDecoderModel.from_pretrained(paths.DONUT_WEIGHTS, config=config)

        # Params for Transformer Decoder
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        # set <s_synthdog> token=57524
        self.model.config.decoder_start_token_id = 57524

    def forward(self, x, labels):

        outputs = self.model(x, labels=labels)    
        return outputs, outputs.loss
    
    def inference(self, x):

        batch_size = x.shape[0]

        decoder_input_ids = torch.full(
            (batch_size, 1), 
            self.model.config.decoder_start_token_id, 
            device=x.device
        )

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                x,
                decoder_input_ids=decoder_input_ids,
                max_length=self.MAX_LENGTH,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        return predictions
