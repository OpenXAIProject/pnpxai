import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './NetworkGraph.css';
import { Box } from '@mui/material';

interface Node {
  id: string;
  children?: Node[];
  // Add any other properties that your nodes might have
}

interface Link {
  source: string;
  target: string;
  // Add any other properties that your links might have
}

interface NetworkGraphProps {
  nodes: Node[];
  links: Link[];
}

const width = 1200;
const height = 800;

const TreeGraph: React.FC<NetworkGraphProps> = ({ nodes, links }) => {
    const d3Container = useRef<SVGSVGElement | null>(null);

    useEffect(() => {
      if (nodes.length && links.length && d3Container.current) {
        const svg = d3.select(d3Container.current);
        svg.selectAll("*").remove();

        const g = svg.append<SVGSVGElement>('g');

        // Zoom functionality
        const zoom = d3.zoom<SVGSVGElement, unknown>()
          .on('zoom', (event) => {
            g.attr('transform', event.transform);
          });

        // Apply the zoom behavior to the svg element
        svg.call(zoom);

        const root = buildTree(nodes, links);

        const treeLayout = d3.tree<Node>().size([width, height]);
        treeLayout(root);
        
        // Draw links
        // @ts-ignore
        g.selectAll('.link')
          .data(root.links())
          .enter().append('line')
          .classed('link', true)
          .attr('x1', (d : any) => d.source.x ?? 0)
          .attr('y1', (d : any) => d.source.y ?? 0)
          .attr('x2', (d : any) => d.target.x ?? 0)
          .attr('y2', (d : any) => d.target.y ?? 0)
          .attr('stroke', '#555');

        // Draw nodes
        const nodeEnter = g.selectAll('.node')
            .data(root.descendants())
            .enter().append('g')
            .classed('node', true)
            .attr('transform', (d : any) => `translate(${d.x ?? 0},${d.y ?? 0})`);

        nodeEnter.append('circle')
            .attr('r', 5)
            .attr('fill', '#fff')
            .attr('stroke', '#69b3a2')
            .attr('stroke-width', 2);

        // Append hover text elements last so they are on top
        const nodeBox = nodeEnter.append('g')
            .style('opacity', 0); // Initially hidden

        nodeBox.append('text')
            .attr('x', 15)
            .attr('y', 5)
            .text(d => d.data.id)
            .attr('font-size', '15px')
            .attr('fill', 'black')
            .attr('pointer-events', 'none'); // Prevent the text from blocking mouse events for nodes

        nodeEnter.on('mouseover', function () {
            d3.select(this).select('g')
                .style('opacity', 1); // Show the large label on mouseover
        }).on('mouseout', function () {
            d3.select(this).select('g')
                .style('opacity', 0); // Hide the large label on mouseout
        });
      }
    }, [nodes, links]);

    return (
      <Box sx={{ m : 2}}>
        <svg
          className="d3-component"
          width={width} 
          height={height}
          ref={d3Container}
          viewBox={`-50 -50 ${width+100} ${height+100}`}
        />
      </Box>
    );
};

function buildTree(nodes: Node[], links: Link[]): d3.HierarchyNode<Node> {
  const idToNode = new Map<string, Node>();
  nodes.forEach(node => {
    idToNode.set(node.id, { ...node, children: [] });
  });

  links.forEach(link => {
    const parent = idToNode.get(link.source);
    const child = idToNode.get(link.target);
    if (parent && child) {
      parent.children = parent.children || [];
      parent.children.push(child);
    }
  });

  return d3.hierarchy(idToNode.get(nodes[0].id) as Node);
}

export default TreeGraph;