import React from 'react';
import { 
    Box, Snackbar, Alert, AlertTitle, 
    Accordion, AccordionDetails, AccordionSummary
} from '@mui/material';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';

export interface ErrorProps {
    name: string;
    message: string;
    trace?: string;
}
  
export const ErrorSnackbar: React.FC<ErrorProps> = ({ name, message, trace }) => {
  const [open, setOpen] = React.useState(true);

  const handleClose = () => setOpen(false);

  const addTraceTitle = () => {
      if (trace) {
      return (
          <AlertTitle>
          {name}: {message}
          </AlertTitle>
      );
      } else {
      return <AlertTitle>{name}</AlertTitle>;
      }
  };

  const renderTrace = (trace: string) => {
      return trace.slice(0, -2).split('\\n').map((line, index) => {
      let toPrint = line;
      if (index % 3 === 0) {
          toPrint = line.replace("'", "").replace(",", "").replace(" '", "").replace("[", "");
      }

      return (
          <pre key={index}>
          {toPrint}
          </pre>
      );
      });
  };

  return (
      <Snackbar anchorOrigin={{ vertical : 'top', horizontal : 'right' }} open={open} onClose={handleClose}>
      <Alert severity="error" onClose={handleClose}>
          {addTraceTitle()}
          {trace && (
          <Box sx={{ mt : 3 }}>
              <Accordion sx={{ backgroundColor : '#FF9999', boxShadow : 0}}>
              <AccordionSummary >
              <ArrowDropDownIcon />
                  Trace
              </AccordionSummary>
              <AccordionDetails sx={{ maxHeight: '500px', overflow: 'auto' }}>
                  {renderTrace(trace)}
              </AccordionDetails>
              </Accordion>
          </Box>
          )}
      </Alert>
      </Snackbar>
  );
  };
