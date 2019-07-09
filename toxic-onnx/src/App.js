import React from 'react';
import * as tf from '@tensorflow/tfjs';
//import { Tensor, InferenceSession } from "onnxjs";
import '@vkontakte/vkui/dist/vkui.css';
import {CellButton, Cell, Div, FormLayout, Group, InfoRow, Input, List, Panel, PanelHeader, Spinner, View} from '@vkontakte/vkui';
import vc from './vocab2embed.json';
import * as onnx from "tfjs-onnx";


class App extends React.Component {
    constructor() {
        super();

        this.state = {
            text: '',
            tokenizer: null,
            model: null,
            data: [
                0, //toxic
                0, //severe toxic
                0, //obscene
                0, //threat
                0, //insult
                0  //identity hate
            ],
        };

        this.loadModel = this.loadModel.bind(this);
        this.changeText = this.changeText.bind(this);
        this.clearText = this.clearText.bind(this);
        this.calculate = this.calculate.bind(this);

        this.loadModel();
        this.state.tokenizer = this.loadTokenizer();
    }

    async calculate(text) {
        let data = [
            0, //toxic
            0, //severe toxic
            0, //obscene
            0, //threat
            0, //insult
            0  //identity hate
        ];

        if (text && this.state.model) {
            let inputs = await this.tokenize(text);
            //const tensor = new Tensor(inputs, "int32");
            //data = await this.state.model.predict(tensorBuffer.toTensor()).data();
            //const outputMap = await this.state.model.run(tensor);
            const outputMap = await this.state.model.predict(tf.tensor(inputs)).data();
            console.log(outputMap);
            // const outputTensor = outputMap.values().next().value;
            // console.log(outputTensor);
        }

        this.setState({data: data});

    }

    changeText(e) {
        let text = e.target.value.toLowerCase();
        this.setState({text: text});
        this.calculate(text);
    }

    clearText() {
        this.setState({text: ''});
        this.calculate('');
    }

    async tokenize(text){
        ////var vc = {'c@@':7, 'world':4, 'hello':3, 'a':5,};
        const max_len = 200;

        text = text.normalize('NFKD').toLowerCase();
        let tokens = text.match(this.state.tokenizer);
        let ids = [vc['[BOS]']];
        // for (var t in tokens) {
        //     ids.push(vc[t])
        // };
        for (var i=0; i<tokens.length; i++) {
            ids.push(vc[tokens[i]])
        };
        if (ids.length > max_len-1) {
            ids = ids.slice(0, max_len-1)
        };
        ids.push(vc['[EOS]']);
        var t = max_len - ids.length;
        if (t < max_len) {
            for (var j=0; j<t; j++)
            {
                ids.push(vc['[EOS]']);
            }
        };
        console.log(ids);
        return ids;
    }

    async loadModel() {
        //const session = new InferenceSession({backendHint : 'cpu'});
        //let loadedModel = await session.loadModel(process.env.PUBLIC_URL + '/kd_blendcnn_noembed.onnx');
        var session = new onnx.onnx.loadModel(process.env.PUBLIC_URL + '/kd_blendcnn_noembed.onnx');
        this.setState({model: session});
        if (this.state.text) {
            this.calculate(this.state.text);
        }
    }

    loadTokenizer(){
        var sorted = [];
        for (var key in vc) {
            sorted[sorted.length] = key;
        }
        sorted.reverse();

        let bpe = new RegExp(sorted.join('|'), "g");
        //this.setState({tokenizer: bpe}, () => console.log(this.state));
        return bpe;
    }

    static renderScore(score) {
        return (Math.round(score * 1000) / 10) + '%';
    }

    static renderEmoji(score) {
        if (score >= 0.9) {
            return '👿';
        } else if (score >= 0.5) {
            return '😡';
        } else if (score >= 0.2) {
            return '😳';
        } else {
            return '😀'
        }
    }

    render() {
        let content = <Div><Spinner/></Div>;
        if (this.state.model) {
            content = <List>
                <Cell>
                    <InfoRow title="Toxic">
                        {App.renderScore(this.state.data[0])} {App.renderEmoji(this.state.data[0])}
                    </InfoRow>
                </Cell>
                <Cell>
                    <InfoRow title="Severe Toxic">
                        {App.renderScore(this.state.data[1])} {App.renderEmoji(this.state.data[1])}
                    </InfoRow>
                </Cell>
                <Cell>
                    <InfoRow title="Obscene">
                        {App.renderScore(this.state.data[2])} {App.renderEmoji(this.state.data[2])}
                    </InfoRow>
                </Cell>
                <Cell>
                    <InfoRow title="Threat">
                        {App.renderScore(this.state.data[3])} {App.renderEmoji(this.state.data[3])}
                    </InfoRow>
                </Cell>
                <Cell>
                    <InfoRow title="Insult">
                        {App.renderScore(this.state.data[4])} {App.renderEmoji(this.state.data[4])}
                    </InfoRow>
                </Cell>
                <Cell>
                    <InfoRow title="Identity Hate">
                        {App.renderScore(this.state.data[5])} {App.renderEmoji(this.state.data[5])}
                    </InfoRow>
                </Cell>
            </List>;
        }

        return (
            <View activePanel="mainPanel">
                <Panel id="mainPanel">
                    <PanelHeader>The Toxic Detector</PanelHeader>
                    <Group>
                        <FormLayout>
                            <Input type="text" top="Your text"  value={this.state.text} onChange={this.changeText}/>
                            <CellButton level="danger" onClick={this.clearText}>Clear text</CellButton>
                        </FormLayout>
                    </Group>

                    <Group title="Results">
                        {content}
                    </Group>
                </Panel>
            </View>
        );
    }
}

export default App;
