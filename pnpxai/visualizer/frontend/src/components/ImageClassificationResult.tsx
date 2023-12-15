import React from 'react';
import { Card, CardContent, Typography, Box, LinearProgress } from '@mui/material';
import { ImageList, ImageListItem, ImageListItemBar } from '@mui/material';


interface ModelPrediction {
  label: string;
  probability: number;
}

interface ImageClassificationResult {
  imagePath: string;
  trueLabel: string;
  modelPredictions: ModelPrediction[];
  isCorrect: boolean;
}

interface ImageClassificationResultsProps {
  numObjectsinLine: number;
  algorithms: string[];
  results: ImageClassificationResult[];
}

const ImageClassificationResults: React.FC<ImageClassificationResultsProps> = ({ numObjectsinLine, algorithms, results }) => {
  // Add "Original" to the list of algorithms
  algorithms = ['Original', ...algorithms];

  // Function to chunk the array into rows
  const chunkArray = (arr: any[], size: number) =>
    arr.reduce((acc, val, i) => {
      let idx = Math.floor(i / size);
      let page = acc[idx] || (acc[idx] = []);
      page.push(val);
      return acc;
    }, []);

  // Chunk the algorithms array into rows
  const algorithmRows = chunkArray(algorithms, numObjectsinLine);

  return (
    <Box sx={{ m: 1 }}>
      {results.map((result, index) => (
        <Box key={index} sx={{ 
          marginBottom: 4, 
          paddingBottom: 2, 
          borderBottom: '2px solid #e0e0e0', // Adds a bottom border for separation
          '&:last-child': {
            borderBottom: 'none' // Removes the border for the last item
          }
        }}>
          {/* First Line - Info Cards */}
          <Box sx={{ display: 'flex', justifyContent: 'space-around', marginBottom: 2 }}>
            {/* Label Card */}
            <Card sx={{ minWidth: 275 }}>
              <CardContent>
                <Typography variant="h5" component="div">
                  True Label
                </Typography>
                <Typography variant="body2">
                  {result.trueLabel}
                </Typography>
              </CardContent>
            </Card>

            {/* Probability Card */}
            <Card sx={{ minWidth: 275 }}>
              <CardContent>
                <Typography variant="h5" component="div">
                  Probabilities
                </Typography>
                {result.modelPredictions.map((prediction, pIndex) => (
                  <Box key={pIndex} sx={{ mb: 1 }}>
                    <Typography variant="body2">
                      {prediction.label}: {Math.round(prediction.probability * 100)}%
                    </Typography>
                    <LinearProgress variant="determinate" value={prediction.probability * 100} />
                  </Box>
                ))}
              </CardContent>
            </Card>

            {/* Result Card */}
            <Card sx={{ minWidth: 275, bgcolor: result.isCorrect ? 'lightgreen' : 'red' }}>
              <CardContent>
                <Typography variant="h5" component="div">
                  {result.isCorrect ? 'Correct' : 'False'}
                </Typography>
              </CardContent>
            </Card>
          </Box>

          {/* Second Line - Image Card */}
          {/* {algorithmRows.map((row, rowIndex) => (
            <ImageList key={rowIndex} sx={{ width: '100%', height: '300px' }} cols={numObjectsinLine} rowHeight={164}>
              {row.map((algorithm, cardIndex) => (
                <ImageListItem key={cardIndex} sx={{ width: '240px' }}>
                  <img
                    src={result.imagePath}
                    alt={`Image ${cardIndex}`}
                    loading="lazy"
                    style={{ width: '100%', height: '100%' }}
                  />
                  <ImageListItemBar 
                    title={cardIndex !== 0 ? `${algorithm} (Rank ${cardIndex + 1 + rowIndex * numObjectsinLine})` : algorithm} 
                    position="below" 
                    sx={{ textAlign: 'center'}} />
                  {cardIndex !== 0 && (
                    <Box sx={{ p: 1 }}>
                      <Typography variant="body2" sx={{ textAlign: 'center' }}> Faithfulness </Typography>
                      <LinearProgress variant="determinate" value={70} />
                      <Typography variant="body2" sx={{ textAlign: 'center' }}> Robustness </Typography>
                      <LinearProgress variant="determinate" value={30} color="secondary" />
                    </Box>
                  )}
                </ImageListItem>
              ))}
            </ImageList>
          ))} */}
          <ImageList sx={{ width: '100%', height: '300px' }} cols={algorithms.length} rowHeight={164}>
            {algorithms.map((algorithm, cardIndex) => (
            <ImageListItem key={cardIndex+1} sx={{ width: "240px"}}>
              <img
                src={result.imagePath}
                alt={`Image ${cardIndex+1}`}
                loading="lazy"
                style={{ width: '100%', height: '100%' }}
              />
              <ImageListItemBar 
                title={cardIndex !== 0 ? `${algorithm} (Rank ${cardIndex + 1})` : algorithm} 
                position="below" 
                sx={{ textAlign: 'center'}} />
              {cardIndex !== 0 && (
                <Box sx={{ p: 1 }}>
                  <Typography variant="body2" sx={{ textAlign: 'center' }}> Faithfulness </Typography>
                  <LinearProgress variant="determinate" value={70} />
                  <Typography variant="body2" sx={{ textAlign: 'center' }}> Robustness </Typography>
                  <LinearProgress variant="determinate" value={30} color="secondary" />
                </Box>
              )}
            </ImageListItem>
            ))}
          </ImageList>
        </Box>
      ))}
    </Box>
  );
}

export default ImageClassificationResults;


// {/* <Box sx={{ display: 'flex', justifyContent: 'space-around' }}> {/* Flex container for horizontal alignment */}
//   {repeatedCards.map((_, cardIndex) => (
//     <Box key={cardIndex} sx={{ maxWidth: 150, margin: '0 auto' }}>
//       <Card>
//         <CardContent>
//           <Typography variant="body2" component="div" sx={{ textAlign: 'center' }}>
//             {result.trueLabel} {/* Title for the image */}
//           </Typography>
//           <img src={result.imagePath} alt={`Image ${cardIndex}`} style={{ width: '120px', height: '110px' }} />
//         </CardContent>
//       </Card>
//     </Box>
//   ))}
// </Box> */}