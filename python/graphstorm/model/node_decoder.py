"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Node prediction decoders.
"""
import sys

import torch as th
from torch import nn

from .gs_layer import GSLayer

class EntityClassifier(GSLayer):
    ''' Classifier for node entities.

    Parameters
    ----------
    in_dim : int
        The input dimension
    num_classes : int
        The number of classes to predict
    multilabel : bool
        Whether this is multi-label classification.
    dropout : float
        The dropout
    '''
    def __init__(self,
                 in_dim,
                 num_classes,
                 multilabel,
                 use_nesting=False,
                 nesting_list=None,
                 dropout=0):
        super().__init__()
        self._in_dim = in_dim
        self._num_classes = num_classes
        self._multilabel = multilabel
        self._use_nesting = use_nesting
        if use_nesting:
            self.decoder = MRL_Linear_Layer(
                nesting_list=nesting_list,
                num_classes=num_classes,
                efficient=True,
            )
        else:
            self.decoder = nn.Parameter(th.Tensor(in_dim, num_classes))
            nn.init.xavier_uniform_(self.decoder,
                                    gain=nn.init.calculate_gain('relu'))
        # TODO(zhengda): The dropout is not used here.
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        ''' The forward function.

        Parameters
        ----------
        inputs : tensor
            The input features

        Returns
        -------
        Tensor : the logits
        '''
        if self._use_nesting:
            return self.decoder(inputs)
        else:
            return th.matmul(inputs, self.decoder)

    def predict(self, inputs):
        """ Make prediction on input data.

        Parameters
        ----------
        inputs : tensor
            The input features

        Returns
        -------
        Tensor : maximum of the predicted results
        """
        if self._use_nesting:
            logits = self.decoder(inputs)
            logits=th.stack(logits, dim=0)[-1, :, :]
        else:
            logits = th.matmul(inputs, self.decoder)
        return (th.sigmoid(logits) > .5).long() if self._multilabel else logits.argmax(dim=1)

    def predict_proba(self, inputs):
        """ Make prediction on input data.

        Parameters
        ----------
        inputs : tensor
            The input features

        Returns
        -------
        Tensor : all normalized predicted results
        """
        if self._use_nesting:
            logits = self.decoder(inputs)
            logits=th.stack(logits, dim=0)
        else:
            logits = th.matmul(inputs, self.decoder)
        return th.sigmoid(logits) if self._multilabel else th.softmax(logits, 1)

    @property
    def in_dims(self):
        """ The number of input dimensions.
        """
        return self._in_dim

    @property
    def out_dims(self):
        """ The number of output dimensions.
        """
        return self._num_classes

class MRL_Linear_Layer(GSLayer):
    def __init__(
            self,
            nesting_list,
            num_classes,
            efficient=True,
            **kwargs):
        super().__init__()
        self.nesting_list = nesting_list
        self.efficient = efficient
        if self.efficient:
            setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], num_classes, **kwargs))
        else:
            for i, num_feat in enumerate(self.nesting_list):
                setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, num_classes, **kwargs))

    def reset_parameters(self):
        if self.efficient:
            self.nesting_classifier_0.reset_parameters()
        else:
            for i in range(len(self.nesting_list)):
                getattr(self, f"nesting_classifier_{i}").reset_parameters()


    def forward(self, x):
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                if self.nesting_classifier_0.bias is None:
                    nesting_logits += (
                        th.matmul(
                            x[:, :num_feat],
                            (self.nesting_classifier_0.weight[:, :num_feat]).t())
                        )
                else:
                    nesting_logits += (
                        th.matmul(
                            x[:, :num_feat],
                            (self.nesting_classifier_0.weight[:, :num_feat]).t()
                        )
                            + self.nesting_classifier_0.bias,
                    )
            else:
                nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)

        return nesting_logits

class EntityRegression(GSLayer):
    ''' Regression on entity nodes

    Parameters
    ----------
    h_dim : int
        The hidden dimensions
    dropout : float
        The dropout
    '''
    def __init__(self,
                 h_dim,
                 dropout=0):
        super(EntityRegression, self).__init__()
        self.h_dim = h_dim
        self.decoder = nn.Parameter(th.Tensor(h_dim, 1))
        nn.init.xavier_uniform_(self.decoder)
        # TODO(zhengda): The dropout is not used.
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        ''' The forward function.
        '''
        return th.matmul(inputs, self.decoder)

    def predict(self, inputs):
        """ The prediction function.

        Parameters
        ----------
        inputs : tensor
            The input features

        Returns
        -------
        Tensor : the predicted results
        """
        return th.matmul(inputs, self.decoder)

    def predict_proba(self, inputs):
        """ Make prediction on input data.
            For regression task, it is same as predict

        Parameters
        ----------
        inputs : tensor
            The input features

        Returns
        -------
        Tensor : all normalized predicted results
        """
        return self.predict(inputs)

    @property
    def in_dims(self):
        """ The number of input dimensions.
        """
        return self.h_dim

    @property
    def out_dims(self):
        """ The number of output dimensions.
        """
        return 1
