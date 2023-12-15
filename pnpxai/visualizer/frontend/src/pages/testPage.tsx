// src/pages/TestPage.tsx
import React, { useEffect } from 'react';
import { Container, Box, Typography, Grid, Paper } from '@mui/material';
import TreeGraph from '../components/ModelGraph';
import { useSelector } from 'react-redux';
import { RootState } from '../app/store';

const TestPage: React.FC = () => {
  // const projectId = "test_project"; // Replace with your actual project ID
  // const projectData = useSelector((state: RootState) => {
  //   return state.projects.data.find(project => project.id === projectId);
  // });

  // const [nodes, setNodes] = React.useState<any[]>([]);
  // const [links, setLinks] = React.useState<any[]>([]);

  // useEffect(() => {
  //   if (projectData?.experiments[0].model.nodes && projectData?.experiments[0].model.edges) {
  //     const deepCopy = JSON.parse(JSON.stringify(projectData?.experiments[0].model));
  //     console.log(deepCopy);
  //     // setNodes(deepCopy.nodes);
  //     // setLinks(deepCopy.edges);
  //   }
  // }
  // , [projectData]);


//   const nodes = [
//     { id: "node1" }, 
//     { id: "node2" }, 
//     { id: "node3" }, 
//     { id: "node4" }, 
//     { id: "node5" }, 
//     { id: "node6" }, 
//     { id: "node7" }, 
//     { id: "node8" }, 
//     { id: "node9" }, 
//     { id: "node10" }
// ];

// const links = [
//     { source: "node1", target: "node2" },
//     { source: "node2", target: "node3" },
//     { source: "node3", target: "node4" },
//     { source: "node4", target: "node5" },
//     { source: "node5", target: "node6" },
//     { source: "node6", target: "node7" },
//     { source: "node7", target: "node8" },
//     { source: "node8", target: "node9" },
//     { source: "node9", target: "node10" }
// ];

  const mockup = {
    "name": "ResNet",
    "nodes": [
        {
            "id": "x"
        },
        {
            "id": "conv1"
        },
        {
            "id": "bn1"
        },
        {
            "id": "relu"
        },
        {
            "id": "maxpool"
        },
        {
            "id": "add"
        },
        {
            "id": "layer1_0_relu_1"
        },
        {
            "id": "add_1"
        },
        {
            "id": "layer1_1_relu_1"
        },
        {
            "id": "layer2_0_downsample_0"
        },
        {
            "id": "layer2_0_downsample_1"
        },
        {
            "id": "add_2"
        },
        {
            "id": "layer2_0_relu_1"
        },
        {
            "id": "add_3"
        },
        {
            "id": "layer2_1_relu_1"
        },
        {
            "id": "layer3_0_downsample_0"
        },
        {
            "id": "layer3_0_downsample_1"
        },
        {
            "id": "add_4"
        },
        {
            "id": "layer3_0_relu_1"
        },
        {
            "id": "add_5"
        },
        {
            "id": "layer3_1_relu_1"
        },
        {
            "id": "layer4_0_downsample_0"
        },
        {
            "id": "layer4_0_downsample_1"
        },
        {
            "id": "add_6"
        },
        {
            "id": "layer4_0_relu_1"
        },
        {
            "id": "add_7"
        },
        {
            "id": "layer4_1_relu_1"
        },
        {
            "id": "avgpool"
        },
        {
            "id": "flatten"
        },
        {
            "id": "fc"
        },
        {
            "id": "output"
        },
        {
            "id": "layer4_1_conv1"
        },
        {
            "id": "layer4_1_bn1"
        },
        {
            "id": "layer4_1_relu"
        },
        {
            "id": "layer4_1_conv2"
        },
        {
            "id": "layer4_1_bn2"
        },
        {
            "id": "layer4_0_conv1"
        },
        {
            "id": "layer4_0_bn1"
        },
        {
            "id": "layer4_0_relu"
        },
        {
            "id": "layer4_0_conv2"
        },
        {
            "id": "layer4_0_bn2"
        },
        {
            "id": "layer3_1_conv1"
        },
        {
            "id": "layer3_1_bn1"
        },
        {
            "id": "layer3_1_relu"
        },
        {
            "id": "layer3_1_conv2"
        },
        {
            "id": "layer3_1_bn2"
        },
        {
            "id": "layer3_0_conv1"
        },
        {
            "id": "layer3_0_bn1"
        },
        {
            "id": "layer3_0_relu"
        },
        {
            "id": "layer3_0_conv2"
        },
        {
            "id": "layer3_0_bn2"
        },
        {
            "id": "layer2_1_conv1"
        },
        {
            "id": "layer2_1_bn1"
        },
        {
            "id": "layer2_1_relu"
        },
        {
            "id": "layer2_1_conv2"
        },
        {
            "id": "layer2_1_bn2"
        },
        {
            "id": "layer2_0_conv1"
        },
        {
            "id": "layer2_0_bn1"
        },
        {
            "id": "layer2_0_relu"
        },
        {
            "id": "layer2_0_conv2"
        },
        {
            "id": "layer2_0_bn2"
        },
        {
            "id": "layer1_1_conv1"
        },
        {
            "id": "layer1_1_bn1"
        },
        {
            "id": "layer1_1_relu"
        },
        {
            "id": "layer1_1_conv2"
        },
        {
            "id": "layer1_1_bn2"
        },
        {
            "id": "layer1_0_conv1"
        },
        {
            "id": "layer1_0_bn1"
        },
        {
            "id": "layer1_0_relu"
        },
        {
            "id": "layer1_0_conv2"
        },
        {
            "id": "layer1_0_bn2"
        }
    ],
    "edges": [
        {
            "id": "xconv1",
            "source": "x",
            "target": "conv1"
        },
        {
            "id": "conv1bn1",
            "source": "conv1",
            "target": "bn1"
        },
        {
            "id": "bn1relu",
            "source": "bn1",
            "target": "relu"
        },
        {
            "id": "relumaxpool",
            "source": "relu",
            "target": "maxpool"
        },
        {
            "id": "maxpoollayer1_0_conv1",
            "source": "maxpool",
            "target": "layer1_0_conv1"
        },
        {
            "id": "maxpooladd",
            "source": "maxpool",
            "target": "add"
        },
        {
            "id": "addlayer1_0_relu_1",
            "source": "add",
            "target": "layer1_0_relu_1"
        },
        {
            "id": "layer1_0_relu_1layer1_1_conv1",
            "source": "layer1_0_relu_1",
            "target": "layer1_1_conv1"
        },
        {
            "id": "layer1_0_relu_1add_1",
            "source": "layer1_0_relu_1",
            "target": "add_1"
        },
        {
            "id": "add_1layer1_1_relu_1",
            "source": "add_1",
            "target": "layer1_1_relu_1"
        },
        {
            "id": "layer1_1_relu_1layer2_0_conv1",
            "source": "layer1_1_relu_1",
            "target": "layer2_0_conv1"
        },
        {
            "id": "layer1_1_relu_1layer2_0_downsample_0",
            "source": "layer1_1_relu_1",
            "target": "layer2_0_downsample_0"
        },
        {
            "id": "layer2_0_downsample_0layer2_0_downsample_1",
            "source": "layer2_0_downsample_0",
            "target": "layer2_0_downsample_1"
        },
        {
            "id": "layer2_0_downsample_1add_2",
            "source": "layer2_0_downsample_1",
            "target": "add_2"
        },
        {
            "id": "add_2layer2_0_relu_1",
            "source": "add_2",
            "target": "layer2_0_relu_1"
        },
        {
            "id": "layer2_0_relu_1layer2_1_conv1",
            "source": "layer2_0_relu_1",
            "target": "layer2_1_conv1"
        },
        {
            "id": "layer2_0_relu_1add_3",
            "source": "layer2_0_relu_1",
            "target": "add_3"
        },
        {
            "id": "add_3layer2_1_relu_1",
            "source": "add_3",
            "target": "layer2_1_relu_1"
        },
        {
            "id": "layer2_1_relu_1layer3_0_conv1",
            "source": "layer2_1_relu_1",
            "target": "layer3_0_conv1"
        },
        {
            "id": "layer2_1_relu_1layer3_0_downsample_0",
            "source": "layer2_1_relu_1",
            "target": "layer3_0_downsample_0"
        },
        {
            "id": "layer3_0_downsample_0layer3_0_downsample_1",
            "source": "layer3_0_downsample_0",
            "target": "layer3_0_downsample_1"
        },
        {
            "id": "layer3_0_downsample_1add_4",
            "source": "layer3_0_downsample_1",
            "target": "add_4"
        },
        {
            "id": "add_4layer3_0_relu_1",
            "source": "add_4",
            "target": "layer3_0_relu_1"
        },
        {
            "id": "layer3_0_relu_1layer3_1_conv1",
            "source": "layer3_0_relu_1",
            "target": "layer3_1_conv1"
        },
        {
            "id": "layer3_0_relu_1add_5",
            "source": "layer3_0_relu_1",
            "target": "add_5"
        },
        {
            "id": "add_5layer3_1_relu_1",
            "source": "add_5",
            "target": "layer3_1_relu_1"
        },
        {
            "id": "layer3_1_relu_1layer4_0_conv1",
            "source": "layer3_1_relu_1",
            "target": "layer4_0_conv1"
        },
        {
            "id": "layer3_1_relu_1layer4_0_downsample_0",
            "source": "layer3_1_relu_1",
            "target": "layer4_0_downsample_0"
        },
        {
            "id": "layer4_0_downsample_0layer4_0_downsample_1",
            "source": "layer4_0_downsample_0",
            "target": "layer4_0_downsample_1"
        },
        {
            "id": "layer4_0_downsample_1add_6",
            "source": "layer4_0_downsample_1",
            "target": "add_6"
        },
        {
            "id": "add_6layer4_0_relu_1",
            "source": "add_6",
            "target": "layer4_0_relu_1"
        },
        {
            "id": "layer4_0_relu_1layer4_1_conv1",
            "source": "layer4_0_relu_1",
            "target": "layer4_1_conv1"
        },
        {
            "id": "layer4_0_relu_1add_7",
            "source": "layer4_0_relu_1",
            "target": "add_7"
        },
        {
            "id": "add_7layer4_1_relu_1",
            "source": "add_7",
            "target": "layer4_1_relu_1"
        },
        {
            "id": "layer4_1_relu_1avgpool",
            "source": "layer4_1_relu_1",
            "target": "avgpool"
        },
        {
            "id": "avgpoolflatten",
            "source": "avgpool",
            "target": "flatten"
        },
        {
            "id": "flattenfc",
            "source": "flatten",
            "target": "fc"
        },
        {
            "id": "fcoutput",
            "source": "fc",
            "target": "output"
        },
        {
            "id": "layer4_1_conv1layer4_1_bn1",
            "source": "layer4_1_conv1",
            "target": "layer4_1_bn1"
        },
        {
            "id": "layer4_1_bn1layer4_1_relu",
            "source": "layer4_1_bn1",
            "target": "layer4_1_relu"
        },
        {
            "id": "layer4_1_relulayer4_1_conv2",
            "source": "layer4_1_relu",
            "target": "layer4_1_conv2"
        },
        {
            "id": "layer4_1_conv2layer4_1_bn2",
            "source": "layer4_1_conv2",
            "target": "layer4_1_bn2"
        },
        {
            "id": "layer4_1_bn2add_7",
            "source": "layer4_1_bn2",
            "target": "add_7"
        },
        {
            "id": "layer4_0_conv1layer4_0_bn1",
            "source": "layer4_0_conv1",
            "target": "layer4_0_bn1"
        },
        {
            "id": "layer4_0_bn1layer4_0_relu",
            "source": "layer4_0_bn1",
            "target": "layer4_0_relu"
        },
        {
            "id": "layer4_0_relulayer4_0_conv2",
            "source": "layer4_0_relu",
            "target": "layer4_0_conv2"
        },
        {
            "id": "layer4_0_conv2layer4_0_bn2",
            "source": "layer4_0_conv2",
            "target": "layer4_0_bn2"
        },
        {
            "id": "layer4_0_bn2add_6",
            "source": "layer4_0_bn2",
            "target": "add_6"
        },
        {
            "id": "layer3_1_conv1layer3_1_bn1",
            "source": "layer3_1_conv1",
            "target": "layer3_1_bn1"
        },
        {
            "id": "layer3_1_bn1layer3_1_relu",
            "source": "layer3_1_bn1",
            "target": "layer3_1_relu"
        },
        {
            "id": "layer3_1_relulayer3_1_conv2",
            "source": "layer3_1_relu",
            "target": "layer3_1_conv2"
        },
        {
            "id": "layer3_1_conv2layer3_1_bn2",
            "source": "layer3_1_conv2",
            "target": "layer3_1_bn2"
        },
        {
            "id": "layer3_1_bn2add_5",
            "source": "layer3_1_bn2",
            "target": "add_5"
        },
        {
            "id": "layer3_0_conv1layer3_0_bn1",
            "source": "layer3_0_conv1",
            "target": "layer3_0_bn1"
        },
        {
            "id": "layer3_0_bn1layer3_0_relu",
            "source": "layer3_0_bn1",
            "target": "layer3_0_relu"
        },
        {
            "id": "layer3_0_relulayer3_0_conv2",
            "source": "layer3_0_relu",
            "target": "layer3_0_conv2"
        },
        {
            "id": "layer3_0_conv2layer3_0_bn2",
            "source": "layer3_0_conv2",
            "target": "layer3_0_bn2"
        },
        {
            "id": "layer3_0_bn2add_4",
            "source": "layer3_0_bn2",
            "target": "add_4"
        },
        {
            "id": "layer2_1_conv1layer2_1_bn1",
            "source": "layer2_1_conv1",
            "target": "layer2_1_bn1"
        },
        {
            "id": "layer2_1_bn1layer2_1_relu",
            "source": "layer2_1_bn1",
            "target": "layer2_1_relu"
        },
        {
            "id": "layer2_1_relulayer2_1_conv2",
            "source": "layer2_1_relu",
            "target": "layer2_1_conv2"
        },
        {
            "id": "layer2_1_conv2layer2_1_bn2",
            "source": "layer2_1_conv2",
            "target": "layer2_1_bn2"
        },
        {
            "id": "layer2_1_bn2add_3",
            "source": "layer2_1_bn2",
            "target": "add_3"
        },
        {
            "id": "layer2_0_conv1layer2_0_bn1",
            "source": "layer2_0_conv1",
            "target": "layer2_0_bn1"
        },
        {
            "id": "layer2_0_bn1layer2_0_relu",
            "source": "layer2_0_bn1",
            "target": "layer2_0_relu"
        },
        {
            "id": "layer2_0_relulayer2_0_conv2",
            "source": "layer2_0_relu",
            "target": "layer2_0_conv2"
        },
        {
            "id": "layer2_0_conv2layer2_0_bn2",
            "source": "layer2_0_conv2",
            "target": "layer2_0_bn2"
        },
        {
            "id": "layer2_0_bn2add_2",
            "source": "layer2_0_bn2",
            "target": "add_2"
        },
        {
            "id": "layer1_1_conv1layer1_1_bn1",
            "source": "layer1_1_conv1",
            "target": "layer1_1_bn1"
        },
        {
            "id": "layer1_1_bn1layer1_1_relu",
            "source": "layer1_1_bn1",
            "target": "layer1_1_relu"
        },
        {
            "id": "layer1_1_relulayer1_1_conv2",
            "source": "layer1_1_relu",
            "target": "layer1_1_conv2"
        },
        {
            "id": "layer1_1_conv2layer1_1_bn2",
            "source": "layer1_1_conv2",
            "target": "layer1_1_bn2"
        },
        {
            "id": "layer1_1_bn2add_1",
            "source": "layer1_1_bn2",
            "target": "add_1"
        },
        {
            "id": "layer1_0_conv1layer1_0_bn1",
            "source": "layer1_0_conv1",
            "target": "layer1_0_bn1"
        },
        {
            "id": "layer1_0_bn1layer1_0_relu",
            "source": "layer1_0_bn1",
            "target": "layer1_0_relu"
        },
        {
            "id": "layer1_0_relulayer1_0_conv2",
            "source": "layer1_0_relu",
            "target": "layer1_0_conv2"
        },
        {
            "id": "layer1_0_conv2layer1_0_bn2",
            "source": "layer1_0_conv2",
            "target": "layer1_0_bn2"
        },
        {
            "id": "layer1_0_bn2add",
            "source": "layer1_0_bn2",
            "target": "add"
        }
    ]
}

  return (
    <Container maxWidth="lg">
      <Box sx={{ m:2 }}>
        <Typography variant='h1'> Test Page </Typography>
      </Box>
      <Box sx={{ m:2 }}>
        <Typography variant='h2'> Object From Backend </Typography>
        <TreeGraph nodes={mockup.nodes} links={mockup.edges}/>
        {/* {projectData?.experiments[0].model.nodes && projectData?.experiments[0].model.edges && (
          <TreeGraph nodes={nodes} links={links}/>
        )} */}

      </Box>
    </Container>
  )
};

export default TestPage;

